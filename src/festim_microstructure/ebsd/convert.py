"""Import CTF grains with UPXO/DefDAP and preserve scientific diagnostics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from festim_microstructure.ebsd.diagnostics import (
    QualityPanels,
    SegmentationError,
    format_report,
    segmentation_error,
    verify_readback,
    write_orientation_png,
    write_quality_png,
)
from festim_microstructure.ebsd.diagnostics import write_png as write_segerr_png
from festim_microstructure.ebsd.orientation import (
    ROT_X_180,
    euler_bunge_to_quat,
    qconj,
    qmul,
)
from festim_microstructure.ebsd.settings import (
    CUBIC_LAUE,
    Settings,
    settings_from_provenance,
)
from festim_microstructure.formats.ctf import CtfMap
from festim_microstructure.formats.ebsd import EbsdData, read_ebsd, write_ebsd
from festim_microstructure.formats.provenance import write_provenance
from festim_microstructure.meshing.upxo import UPXO_REVISION, import_ebsd

__all__ = [
    "ConversionResult",
    "CtfConversion",
    "MeasureOptions",
    "build_grid",
    "convert",
    "crop_grid",
    "measure_against_ctf",
]


@dataclass
class ConversionResult:
    """Native archive, source provenance, and orientation error populations."""

    archive: Path
    provenance: Path
    settings: Settings
    data: EbsdData
    window: tuple
    segmentation: SegmentationError

    @property
    def rms_deg(self):
        return self.segmentation.indexed.rms

    @property
    def ncells(self):
        return self.data.ncells

    @property
    def cellids(self):
        return self.data.labels

    @property
    def voxsize(self):
        return self.data.vox

    @property
    def extent(self):
        return self.data.extent


@dataclass
class MeasureOptions:
    """Compare source CTF orientations with native grain and/or pixel data."""

    settings: Settings | None = None
    provenance: str | None = None
    against: str = "grain"
    png: str | None = None
    csv: str | None = None

    def resolve(self, ctf_path):
        if self.against not in ("grain", "voxel", "both"):
            raise ValueError("against must be grain, voxel, or both")
        return (
            settings_from_provenance(self.provenance)
            if self.provenance
            else self.settings or Settings(ctf=str(ctf_path))
        )


class CtfConversion:
    """Quality/crop adapter, UPXO import, native persistence and diagnostics.

    Grain detection and symmetry-aware grain means are performed by DefDAP
    through UPXO's modern EBSDReader. No local grain segmentation or topology
    repair is applied. Optional nearest-grain filling is explicit geometry
    preparation; the original membership remains available in the archive.
    """

    def __init__(self, settings, log=print):
        self.settings = settings
        self.log = log or (lambda *args: None)

    def read(self):
        opt = self.settings
        opt.__post_init__()
        self.ctf = CtfMap(opt.ctf)
        self.crysym, phase = self.ctf.crysym(opt.phase)
        if phase is None or phase["laue"] not in CUBIC_LAUE:
            raise ValueError(
                f"Laue group {phase['laue'] if phase else 'unknown'}: "
                "EBSD boundary disorientations currently require m-3m (Laue 11)"
            )
        self.qgrid, self.ok, self.diag = build_grid(
            self.ctf,
            opt.phase,
            opt.max_mad,
            not opt.allow_error,
            opt.min_bands,
            opt.euler_correction_quat,
        )
        self.window = (slice(0, self.ok.shape[0]), slice(0, self.ok.shape[1]))
        if opt.crop:
            self.qgrid, self.ok, self.window = crop_grid(
                self.qgrid,
                self.ok,
                opt.crop,
                self.ctf.header["XStep"],
                self.ctf.header["YStep"],
            )
            self.diag = {k: v[self.window] for k, v in self.diag.items()}
        if not self.ok.any():
            raise ValueError("no indexed pixels survived the quality filters and crop")
        self.vox = (
            self.ctf.header["XStep"] * opt.scale,
            self.ctf.header["YStep"] * opt.scale,
        )
        self.log(
            f"CTF quality/crop: {self.ok.shape[1]} x {self.ok.shape[0]}, "
            f"{self.ok.sum()}/{self.ok.size} indexed pixels"
        )
        return self

    def import_grains(self):
        opt = self.settings
        imported = import_ebsd(self.ctf, self.window, self.ok, opt, log=self.log)
        source = imported["source_labels"]
        labels = np.maximum(source, 0).astype(np.int32)
        if not np.any(labels):
            raise ValueError(
                "no grains survived DefDAP grain detection; lower min_pixels"
            )
        if opt.fill and np.any(labels == 0):
            from scipy.ndimage import distance_transform_edt

            nearest = distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            labels[labels == 0] = labels[tuple(nearest[:, labels == 0])]
        # DefDAP quaternions have the opposite vector sign to our Bunge
        # representation. Conjugate both pixel and symmetry-aware grain means.
        pixels = qconj(imported["pixel_quats"])
        grains = qconj(imported["grain_quats"])
        if opt.euler_correction_quat is not None:
            pixels = qmul(opt.euler_correction_quat, pixels)
            grains = qmul(opt.euler_correction_quat, grains)
        # The CTF parser independently checks transcription/convention; it does
        # not supply the data used for grain detection or the grain means.
        from festim_microstructure.ebsd.diagnostics import theta_field

        transcription = theta_field(self.qgrid, pixels)
        if np.max(transcription[self.ok]) > 1e-3:
            raise ValueError("UPXO import orientations disagree with source CTF")
        self.packages = imported["packages"]
        self.cellids, self.source_labels, self.qcell = labels, source, grains
        self.unassigned = source <= 0
        self.ncells = len(grains)
        self.log(
            f"UPXO/DefDAP: {self.ncells} grains; {self.unassigned.sum()} "
            "pixels without original grain membership"
        )
        self.data = EbsdData(labels, grains, self.vox, pixels, self.ok, source)
        return self

    def measure(self):
        self.segmentation = segmentation_error(
            self.qgrid,
            self.data.labels,
            self.data.grain_quats,
            self.vox,
            ok=self.ok,
            threshold=self.settings.threshold,
            backfilled=self.data.source_labels <= 0,
            qvox=self.data.pixel_quats,
        )
        for line in format_report(self.segmentation):
            self.log("  " + line)
        return self

    def write(self, output=None):
        self.archive_path = Path(output or Path(self.settings.ctf).with_suffix(".npz"))
        if self.archive_path.suffix != ".npz":
            raise ValueError(
                "EBSD imports are native .npz archives; use a .npz output path"
            )
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.provenance_path = self.archive_path.with_name(
            self.archive_path.stem + "-provenance.json"
        )
        data = self.data
        if self.settings.flip_y:
            data = EbsdData(
                data.labels[::-1],
                qmul(ROT_X_180, data.grain_quats),
                data.vox,
                qmul(ROT_X_180, data.pixel_quats[::-1]),
                data.indexed[::-1],
                data.source_labels[::-1],
            )
        # Publish only an archive that can be re-read and independently checked.
        with tempfile.TemporaryDirectory(dir=self.archive_path.parent) as scratch:
            candidate = Path(scratch) / "ebsd.npz"
            write_ebsd(candidate, data)
            verify_readback(
                candidate, self.qgrid, self.ok, self.cellids, self.settings.flip_y
            )
            candidate.replace(self.archive_path)
        self.output_data = data
        write_provenance(
            self.provenance_path,
            self.settings,
            self.segmentation,
            self.vox,
            log=self.log,
        )
        rec = json.loads(self.provenance_path.read_text())
        rec.update(
            identity=self.identity(),
            archive_sha256=hashlib.sha256(self.archive_path.read_bytes()).hexdigest(),
            packages=self.packages,
            importer="upxo.interfaces.defdap.ebsd_reader.EBSDReader",
            original_membership_pixels=int((self.source_labels > 0).sum()),
            pruned_pixels=int((self.source_labels == -2).sum()),
            unindexed_pixels=int((self.source_labels == 0).sum()),
        )
        self.provenance_path.write_text(json.dumps(rec, indent=2) + "\n")
        return self

    def identity(self):
        from festim_microstructure.meshing import upxo

        return json.loads(
            json.dumps(
                dict(
                    settings={**asdict(self.settings), "ctf": str(self.settings.ctf)},
                    source_sha256=hashlib.sha256(
                        Path(self.settings.ctf).read_bytes()
                    ).hexdigest(),
                    upxo_revision=UPXO_REVISION,
                    code_sha256=hashlib.sha256(
                        Path(__file__).read_bytes() + Path(upxo.__file__).read_bytes()
                    ).hexdigest(),
                )
            )
        )

    def diagnose(self):
        if self.settings.diagnostics:
            stem = self.archive_path.with_suffix("")
            write_segerr_png(
                f"{stem}-segerr.png",
                self.segmentation,
                self.cellids,
                unit=self.settings.unit,
                flip_y=self.settings.flip_y,
                log=self.log,
            )
            write_quality_png(
                f"{stem}-quality.png",
                QualityPanels(
                    self.diag,
                    self.ok,
                    self.unassigned,
                    self.cellids,
                    self.settings,
                    self.vox,
                    self.settings.unit,
                    self.settings.flip_y,
                ),
                log=self.log,
            )
            write_orientation_png(
                f"{stem}-ipfz.png",
                self.output_data,
                log=self.log,
                unit=self.settings.unit,
            )
        return self

    def result(self):
        return ConversionResult(
            self.archive_path,
            self.provenance_path,
            self.settings,
            self.output_data,
            self.window,
            self.segmentation,
        )

    def run(self, output=None, force=True):
        archive = Path(output or Path(self.settings.ctf).with_suffix(".npz"))
        if archive.suffix != ".npz":
            raise ValueError("EBSD imports require a native .npz output path")
        self.read()
        provenance = archive.with_name(archive.stem + "-provenance.json")
        if not force and archive.is_file() and provenance.is_file():
            try:
                saved = json.loads(provenance.read_text())
            except (ValueError, OSError):
                saved = {}
            if (
                saved.get("identity") == self.identity()
                and saved.get("archive_sha256")
                == hashlib.sha256(archive.read_bytes()).hexdigest()
            ):
                data = read_ebsd(archive)
                self.output_data = data
                if self.settings.flip_y:
                    data = EbsdData(
                        data.labels[::-1],
                        qmul(qconj(ROT_X_180), data.grain_quats),
                        data.vox,
                        qmul(qconj(ROT_X_180), data.pixel_quats[::-1]),
                        data.indexed[::-1],
                        data.source_labels[::-1],
                    )
                self.data, self.cellids, self.source_labels = (
                    data,
                    data.labels,
                    data.source_labels,
                )
                self.unassigned = data.source_labels <= 0
                self.archive_path, self.provenance_path = archive, provenance
                return self.measure().diagnose().result()
        return self.import_grains().measure().write(output).diagnose().result()


def convert(
    ctf_path=None, output=None, *, settings=None, log=print, force=True, **kwargs
):
    """Import a CTF into a native archive through UPXO/DefDAP.

    Quality filters and crop apply before grain detection. Rejected and pruned
    pixels never determine grain means, even when filling includes them in the
    geometry. The archive retains their original status for later diagnostics.
    """
    if kwargs and settings is not None:
        raise TypeError("pass either a Settings object or keyword arguments")
    if settings is None and ctf_path is None:
        raise TypeError("pass ctf_path or a Settings object")
    return CtfConversion(settings or Settings(ctf=str(ctf_path), **kwargs), log).run(
        output, force=force
    )


def measure_against_ctf(ctf_path, archive_path, options=None, log=print):
    """Independently compare persisted grain/pixel orientations with source CTF.

    Original membership is persisted, so the indexed and backfilled populations
    exactly match conversion diagnostics, including pruned pixels.
    """
    opt = options or MeasureOptions(
        provenance=str(
            Path(archive_path).with_name(Path(archive_path).stem + "-provenance.json")
        )
    )
    settings = opt.resolve(ctf_path)
    source = CtfConversion(settings, log=None).read()
    data = read_ebsd(archive_path)
    if data.labels.shape != source.ok.shape:
        raise ValueError("archive shape differs from the source CTF crop")
    if settings.flip_y:
        cells, original = data.labels[::-1], data.source_labels[::-1]
        grains = qmul(qconj(ROT_X_180), data.grain_quats)
        pixels = qmul(qconj(ROT_X_180), data.pixel_quats[::-1])
    else:
        cells, original, grains, pixels = (
            data.labels,
            data.source_labels,
            data.grain_quats,
            data.pixel_quats,
        )
    res = segmentation_error(
        source.qgrid,
        cells,
        grains,
        data.vox,
        ok=source.ok,
        threshold=settings.threshold,
        backfilled=original <= 0,
        qvox=pixels if opt.against in ("voxel", "both") else None,
    )
    if log:
        for line in format_report(res):
            log("  " + line)
    if opt.png:
        write_segerr_png(opt.png, res, cells, unit=settings.unit, log=log)
    if opt.csv:
        from festim_microstructure.ebsd.diagnostics import write_csv

        write_csv(opt.csv, res, log=log)
    return res


def build_grid(
    ctf, phase, max_mad, require_zero_error, min_bands, euler_correction=None
):
    """Place the pixel table on the (ny, nx) grid and build the quality mask.

    Points are indexed from their X/Y coordinates rather than from row order,
    so a file that is not written in strict raster order still lands correctly
    and a truncated file leaves holes rather than shearing the map.
    ``euler_correction`` is an optional quaternion taking the Euler-angle frame
    onto the map frame, left-multiplied onto every orientation.
    """
    ny, nx = ctf.shape
    ix = np.rint(ctf["X"] / ctf.header["XStep"]).astype(int)
    iy = np.rint(ctf["Y"] / ctf.header["YStep"]).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not inside.all():
        print(f"  warning: {int((~inside).sum())} points fall outside XCells x YCells")

    good = inside & (ctf["Phase"] == phase)
    if require_zero_error and ctf.has("Error"):
        good &= ctf["Error"] == 0
    if ctf.has("MAD"):
        good &= ctf["MAD"] <= max_mad
    if min_bands and ctf.has("Bands"):
        good &= ctf["Bands"] >= min_bands

    euler = np.stack((ctf["Euler1"], ctf["Euler2"], ctf["Euler3"]), axis=-1)
    finite = np.isfinite(euler).all(axis=1)
    good &= finite
    euler[~finite] = 0.0
    quat = euler_bunge_to_quat(euler[:, 0], euler[:, 1], euler[:, 2])
    if euler_correction is not None:
        quat = qmul(np.asarray(euler_correction, dtype=float), quat)

    qgrid = np.zeros((ny, nx, 4))
    qgrid[..., 0] = 1.0
    ok = np.zeros((ny, nx), dtype=bool)
    qgrid[iy[inside], ix[inside]] = quat[inside]
    ok[iy[good], ix[good]] = True

    # Preserve per-pixel quality fields for diagnostics.
    diag = {}
    for col, fill in (("Error", -1), ("MAD", np.nan), ("Bands", -1), ("Phase", -1)):
        if ctf.has(col):
            g = np.full((ny, nx), fill, dtype=float)
            g[iy[inside], ix[inside]] = ctf[col][inside]
            diag[col] = g
    return qgrid, ok, diag


def crop_grid(qgrid, ok, spec, xstep, ystep):
    """Crop source coordinates before grain detection and rebase to zero."""
    try:
        x0, x1, y0, y1 = (float(v) for v in spec.split(","))
    except ValueError:
        raise ValueError(
            f"crop={spec!r}: expected four comma-separated numbers, "
            "xmin,xmax,ymin,ymax, in the same units as XStep"
        )
    ny, nx = ok.shape
    ix0, ix1 = max(round(x0 / xstep), 0), min(round(x1 / xstep), nx)
    iy0, iy1 = max(round(y0 / ystep), 0), min(round(y1 / ystep), ny)
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        raise ValueError(
            f"crop={spec} keeps {max(ix1 - ix0, 0)} x {max(iy1 - iy0, 0)} "
            f"pixels. The map is {nx} x {ny} pixels of {xstep} x {ystep}, "
            f"i.e. {nx * xstep:g} x {ny * ystep:g} in those units."
        )
    window = (slice(iy0, iy1), slice(ix0, ix1))
    return qgrid[window], ok[window], window
