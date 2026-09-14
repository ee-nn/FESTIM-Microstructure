"""Drive a ``.ctf`` to ``.tesr`` conversion, stage by stage.

:class:`CtfConversion` holds the state one conversion accumulates and exposes
each stage as its own method, so a caller can stop after the segmentation, look
at it, and only then write. :func:`convert` runs the lot and is what most
callers want.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..orientation import (
    cubic_symmetry_quaternions,
    quat_to_rodrigues,
    rodrigues_to_quat,
    self_test,
    to_fundamental_zone,
)
from ..segmentation_error import (
    SegmentationError,
    format_report,
    read_tesr_full,
    segmentation_error,
)
from ..segmentation_error import write_png as write_segerr_png
from .diagnostics import (
    QualityPanels,
    render_checks,
    verify_readback,
    write_quality_png,
)
from .morphology import fill_holes, make_meshable
from .provenance import settings_from_provenance, write_provenance
from .reader import CtfMap, build_grid, crop_grid
from .segmentation import grain_mean_orientations, relabel_and_prune, segment_grains
from .settings import CUBIC_LAUE, Settings
from .tesr import TesrData, write_tesr

__all__ = [
    "ConversionResult",
    "CtfConversion",
    "MeasureOptions",
    "convert",
    "measure_tesr_against_ctf",
]


@dataclass
class ConversionResult:
    """What one conversion produced.

    ``cellids`` and ``ok`` are in the *written* row order, i.e. after
    ``flip_y``; ``segmentation`` was measured before it, in the .ctf's own
    frame, because neither the mirror nor the active/passive flip changes a
    disorientation.
    """

    tesr: Path
    provenance: Path
    settings: Settings
    crysym: str
    ncells: int
    voxsize: tuple
    extent: tuple  #: (lx, ly) in the .ctf's own length unit, times settings.scale
    cellids: Any
    ok: Any
    ori_cell: Any  #: (ncell, 3) Rodrigues
    qcell: Any  #: (ncell, 4) quaternions
    npx: Any  #: voxels per grain, id order
    window: tuple  #: the crop applied to the .ctf grid
    segmentation: SegmentationError

    @property
    def rms_deg(self) -> float:
        """The headline number: RMS disorientation over the indexed voxels."""
        return self.segmentation.indexed.rms


@dataclass
class MeasureOptions:
    """Inputs for :func:`measure_tesr_against_ctf`.

    Give ``settings`` or ``provenance``: the crop window, the mirror and the
    orientation convention are choices recorded in one of the two and are not
    recoverable from the .tesr alone.
    """

    settings: Settings | None = None
    provenance: str | None = None
    against: str = "grain"
    """``"grain"`` compares each pixel with its grain's ``**cell/*ori``, which
    is the segmentation error and the only orientation the rest of the pipeline
    sees. ``"voxel"`` compares it with its own ``**oridata`` entry, which
    measures transcription only and should come back at the 1e-6 deg noise
    floor. ``"both"`` reports each."""
    png: str | None = None
    csv: str | None = None

    def resolve(self, ctf_path) -> Settings:
        if self.provenance is not None:
            return settings_from_provenance(self.provenance)
        return self.settings or Settings(ctf=str(ctf_path))


class CtfConversion:
    """One ``.ctf`` on its way to a ``.tesr``.

    The stages run in order and each one leaves its output on the instance:
    :meth:`read`, :meth:`segment`, :meth:`clean`, :meth:`orient`,
    :meth:`measure`, :meth:`write`, :meth:`diagnose`. :meth:`run` chains them.
    """

    def __init__(self, settings: Settings, log=print):
        self.settings = settings
        self.log = log or (lambda *a, **k: None)
        self.sym = cubic_symmetry_quaternions()
        self.window: tuple = ()

    # -- stage 1: read and mask -------------------------------------------
    def read(self):
        """Parse the .ctf, build the orientation grid and the quality mask."""
        opt, log = self.settings, self.log
        self.ctf = ctf = CtfMap(opt.ctf)
        ny, nx = ctf.shape
        log(
            f"{opt.ctf}: {nx} x {ny} pixels, step {ctf.header['XStep']} x "
            f"{ctf.header['YStep']}, {ctf.npoints} rows"
        )
        if not np.isclose(ctf.header["XStep"], ctf.header["YStep"]):
            log("  note: XStep != YStep -- voxels will not be square")

        self.crysym, phase = ctf.crysym(opt.phase)
        if self.crysym is None:
            raise ValueError("no phase line found; cannot determine crystal symmetry")
        if phase["laue"] not in CUBIC_LAUE:
            raise ValueError(
                f"phase '{phase['name']}' has Laue group {phase['laue']} -> "
                f"{self.crysym}, which this script cannot segment. The "
                "disorientation used here is specific to the cubic group. "
                "Segment in MTEX and write the grain ids into the **data "
                "section instead."
            )
        log(
            f"  phase {opt.phase}: {phase['name']}, Laue {phase['laue']} -> "
            f"{self.crysym}"
        )

        self.qgrid, self.ok, self.diag = build_grid(
            ctf, opt.phase, opt.max_mad, not opt.allow_error, opt.min_bands
        )
        self.window = (slice(0, ny), slice(0, nx))
        if opt.crop:
            self.qgrid, self.ok, self.window = crop_grid(
                self.qgrid, self.ok, opt.crop, ctf.header["XStep"], ctf.header["YStep"]
            )
            self.diag = {k: v[self.window] for k, v in self.diag.items()}
            ny, nx = self.ok.shape
            log(f"  cropped to {nx} x {ny} pixels ({opt.crop})")
        self.shape = (ny, nx)

        frac = self.ok.mean()
        log(
            f"indexed & above quality cutoffs: {self.ok.sum()} of "
            f"{self.ok.size} ({100 * frac:.1f}%)"
        )
        if frac < 0.8:
            log(
                "  WARNING: a large fraction of the map was rejected. Loosen "
                "max_mad or set allow_error, or expect a holed tessellation"
            )
        return self

    # -- stage 2: grains ---------------------------------------------------
    def segment(self):
        """Flood-fill the pixels into grains and prune the tiny ones."""
        opt, log = self.settings, self.log
        labels = segment_grains(self.qgrid, self.ok, opt.threshold, self.sym)
        self.cellids, self.ncells, dropped, lost = relabel_and_prune(
            labels, self.ok, opt.min_pixels
        )
        log(
            f"  grains at {opt.threshold:g} deg: {self.ncells} "
            f"({dropped} below {opt.min_pixels} px dropped, {lost} px)"
        )
        if self.ncells == 0:
            raise ValueError("no grains survived; lower min_pixels or threshold")
        return self

    # -- stage 3: raster clean-up -----------------------------------------
    def clean(self):
        """Back-fill the holes and remove the topologies ``neper -M`` rejects.

        Degenerate cells are what abort ``neper -T -n from_morpho`` partway
        through "Listing cell voxels", and the objective function cannot place
        ``pts(res=N)`` control points on a cell two pixels across either. The
        bottom of the size distribution is reported so the failure is visible
        here, not there.
        """
        opt, log = self.settings, self.log
        self.unassigned = self.cellids == 0
        empty_before = int(self.unassigned.sum())
        log(
            f"  unassigned voxels: {empty_before} of {self.cellids.size} "
            f"({100 * empty_before / self.cellids.size:.1f} %)"
        )
        if opt.fill:
            self.cellids, filled = fill_holes(self.cellids)
            log(f"  filled {filled} voxels from the nearest cell")
        elif empty_before > 0.05 * self.cellids.size:
            log(
                "  WARNING: the raster has substantial holes and fill=False was "
                "given. The tessellation fit will treat hole boundaries as grain "
                "boundaries"
            )

        if opt.topology_fix:
            self.cellids, absorbed, pinches = make_meshable(self.cellids)
            self.ncells = int(self.cellids.max())
            if absorbed:
                log(
                    f"  absorbed {len(absorbed)} enclosed grain(s) into their "
                    f"surrounding grain: {', '.join(map(str, absorbed))}"
                )
            if pinches:
                log(f"  unpinched {pinches} corner-only self-contact(s), 1 px each")
            if absorbed or pinches:
                log(f"  grains: {self.ncells}")

        self.npx = np.bincount(self.cellids.ravel())[1:]
        log(
            "  smallest grains (px): "
            + ", ".join(str(c) for c in np.sort(self.npx)[:8])
        )
        if self.npx.min() < 10:
            log(
                f"  WARNING: {int((self.npx < 10).sum())} grains under 10 px. "
                "Neper's tessellation fit is liable to abort on these -- raise "
                "min_pixels (20 is a reasonable floor)"
            )
        return self

    # -- stage 4: orientations --------------------------------------------
    def orient(self):
        """One orientation per grain, plus the per-voxel field if requested."""
        opt = self.settings
        ny, nx = self.shape
        self.qfz = to_fundamental_zone(self.qgrid.reshape(-1, 4), self.sym).reshape(
            ny, nx, 4
        )
        self.qcell = grain_mean_orientations(
            self.qgrid, self.cellids, self.ncells, self.sym
        )
        self.ori_cell = quat_to_rodrigues(self.qcell)
        self.ori_vox = None if not opt.voxel_ori else quat_to_rodrigues(self.qfz)
        self.vox = (
            self.ctf.header["XStep"] * opt.scale,
            self.ctf.header["YStep"] * opt.scale,
        )
        return self

    # -- stage 5: what the segmentation cost -------------------------------
    def measure(self):
        """Measure the segmentation error, in the .ctf's own frame.

        Computed before ``active`` flips the sign and ``flip_y`` mirrors the
        arrays; neither transformation changes a disorientation. This is the
        headline quality number for stage 1 of the pipeline: the RMS angle
        between a pixel's measured orientation and the single orientation its
        grain will carry from here on.
        """
        self.segmentation = segmentation_error(
            self.qgrid,
            self.cellids,
            self.qcell,
            self.vox,
            ok=self.ok,
            threshold=self.settings.threshold,
            backfilled=self.unassigned,
        )
        for line in format_report(self.segmentation):
            self.log("  " + line)
        return self

    # -- stage 6: write ----------------------------------------------------
    def write(self, output=None):
        """Apply the orientation convention and the mirror, then write."""
        opt, log = self.settings, self.log
        if opt.active:  # active is the opposite rotation, i.e. -r
            self.ori_cell = -self.ori_cell
            if self.ori_vox is not None:
                self.ori_vox = -self.ori_vox

        if opt.flip_y:
            self.cellids = self.cellids[::-1]
            self.ok = self.ok[::-1]
            if self.ori_vox is not None:
                self.ori_vox = self.ori_vox[::-1]

        self.tesr_path = Path(output or Path(opt.ctf).with_suffix(".tesr"))
        write_tesr(
            self.tesr_path,
            TesrData(
                cellids=self.cellids,
                ori_cell=self.ori_cell,
                voxsize=self.vox,
                crysym=self.crysym,
                ori_vox=self.ori_vox,
                oridef=self.ok,
            ),
        )
        self.provenance_path = self.tesr_path.with_name(
            self.tesr_path.stem + "-provenance.json"
        )
        write_provenance(
            self.provenance_path, opt, self.segmentation, self.vox, log=log
        )
        return self

    # -- stage 7: diagnostics ----------------------------------------------
    def diagnose(self):
        """The optional check images and the read-back verification."""
        opt, log = self.settings, self.log
        if not opt.diagnostics:
            return self
        unit = opt.unit
        write_segerr_png(
            self.tesr_path.with_name(self.tesr_path.stem + "-segerror.png"),
            self.segmentation,
            # back to the .ctf's row order: `cellids` was mirrored in place
            # above, while the theta field was measured before that
            self.cellids[::-1] if opt.flip_y else self.cellids,
            unit=unit,
            log=log,
        )
        write_quality_png(
            self.tesr_path.with_name(self.tesr_path.stem + "-quality.png"),
            QualityPanels(
                diag=self.diag,
                ok=self.ok,
                unassigned=self.unassigned,
                cellids=self.cellids,
                settings=opt,
                vox=self.vox,
                unit=unit,
                flip_y=opt.flip_y,
            ),
            log=log,
        )
        for line in verify_readback(
            self.tesr_path, self.qgrid, self.ok, self.cellids, opt.flip_y
        ):
            log(f"  {line}")
        if opt.neper:
            render_checks(
                self.tesr_path,
                self.extent[0],
                unit=unit,
                neper=opt.neper,
                povray=opt.povray,
                log=log,
            )
        return self

    @property
    def extent(self):
        ny, nx = self.shape
        return nx * self.vox[0], ny * self.vox[1]

    def result(self) -> ConversionResult:
        return ConversionResult(
            tesr=self.tesr_path,
            provenance=self.provenance_path,
            settings=self.settings,
            crysym=self.crysym,
            ncells=self.ncells,
            voxsize=self.vox,
            extent=self.extent,
            cellids=self.cellids,
            ok=self.ok,
            ori_cell=self.ori_cell,
            qcell=self.qcell,
            npx=self.npx,
            window=self.window,
            segmentation=self.segmentation,
        )

    def run(self, output=None) -> ConversionResult:
        self_test()
        self.read().segment().clean().orient().measure().write(output).diagnose()
        self._summarise()
        return self.result()

    def _summarise(self):
        opt, log = self.settings, self.log
        lx, ly = self.extent
        grain_size = np.sqrt(self.npx.mean()) * self.vox[0]
        out = self.tesr_path
        log(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        log(f"  domain      : {lx:.4g} x {ly:.4g}")
        log(f"  grain size  : ~{grain_size:.4g} (equivalent square)")
        if not opt.diagnostics:
            log(f"  set diagnostics=True for {out.stem}-quality.png and the rest")
        if self.ncells > 400:
            side = grain_size * np.sqrt(250)
            log(
                f"\n{self.ncells} grains is a long tessellation fit. Crop to ~250 "
                f"by re-running with a window about {side:.3g} x {side:.3g} in "
                f'the .ctf\'s units, i.e. crop="x0,x0+{side:.3g},y0,y0+{side:.3g}"'
            )


def convert(ctf_path=None, output=None, *, settings=None, log=print, **kwargs):
    """Convert a .ctf into a .tesr, and measure what the segmentation cost.

    Either give `ctf_path` plus any `Settings` field as a keyword argument, or
    build a `Settings` yourself and pass it as `settings`. `output` defaults to
    the .ctf's name with a .tesr suffix. `log` takes every progress line and can
    be set to None to run silently.

    Returns a :class:`ConversionResult`; its ``rms_deg`` is the headline RMS
    disorientation in degrees.

    Raises ValueError if the map has no cubic phase line or if no grain
    survives the prune, rather than exiting the interpreter.
    """
    if kwargs and settings is not None:
        raise TypeError("pass either a Settings object or keyword arguments")
    opt = settings if settings is not None else Settings(ctf=ctf_path, **kwargs)
    return CtfConversion(opt, log=log).run(output)


def measure_tesr_against_ctf(ctf_path, tesr_path, options=None, log=print):
    """Re-measure the segmentation error of a .ctf/.tesr pair already on disk.

    `convert` reports this as it goes; this is the same measurement made
    afterwards, straight off the two files, so it also verifies that what was
    written is what was meant. See :class:`MeasureOptions`.

    Returns the :class:`~...segmentation_error.SegmentationError`.
    """
    log = log or (lambda *a, **k: None)
    mopt = options or MeasureOptions()
    opt = mopt.resolve(ctf_path)

    ctf = CtfMap(ctf_path)
    _crysym, phase = ctf.crysym(opt.phase)
    if phase is None or phase["laue"] not in CUBIC_LAUE:
        raise ValueError(
            "cubic phases only: the disorientation used here is cubic-specific"
        )
    qgrid, ok, _diag = build_grid(
        ctf, opt.phase, opt.max_mad, not opt.allow_error, opt.min_bands
    )
    if opt.crop:
        qgrid, ok, _w = crop_grid(
            qgrid, ok, opt.crop, ctf.header["XStep"], ctf.header["YStep"]
        )

    t = read_tesr_full(tesr_path)
    if (t["ny"], t["nx"]) != ok.shape:
        raise ValueError(
            f"{tesr_path} is {t['nx']} x {t['ny']} voxels but the .ctf window "
            f"is {ok.shape[1]} x {ok.shape[0]}. Pass the Settings (or the "
            "provenance json) of the conversion that wrote it."
        )
    cells = t["cells"][::-1] if opt.flip_y else t["cells"]
    sign = -1.0 if opt.active else 1.0
    qcell = rodrigues_to_quat(sign * t["cell_ori"])

    qvox = None
    if mopt.against in ("voxel", "both"):
        if "vox_ori" in t:
            r = t["vox_ori"][::-1] if opt.flip_y else t["vox_ori"]
            qvox = rodrigues_to_quat(sign * r).reshape((*ok.shape, 4))
        else:
            log("  note: no **oridata in the tesr (voxel_ori=False); check skipped")

    res = segmentation_error(
        qgrid, cells, qcell, t["vox"], ok=ok, threshold=opt.threshold, qvox=qvox
    )
    for line in format_report(res):
        log("  " + line)

    # `convert` knows which voxels fill_holes back-filled, including the ones
    # whose grain was pruned by min_pixels; working from the files alone only
    # the quality rejections are visible, so the two populations differ by the
    # pruned pixels and the numbers differ with them. Say so rather than leave
    # two unequal RMS values lying about.
    ref = {}
    if mopt.provenance is not None:
        ref = json.loads(Path(mopt.provenance).read_text())
        ref = ref.get("segmentation_error_deg", {}).get("indexed", {})
    if ref and ref.get("n") != res.indexed.n:
        log(
            f"  note: convert() measured {ref['rms']:.3f} deg over {ref['n']} "
            f"voxels. The {res.indexed.n - ref['n']} extra voxels here "
            "belonged to grains the prune removed and were then back-filled; "
            "only the conversion can tell them apart from voxels in a grain of "
            "their own."
        )

    if mopt.png:
        write_segerr_png(mopt.png, res, cells, unit=opt.unit, log=log)
    if mopt.csv:
        from ..segmentation_error import write_csv

        write_csv(mopt.csv, res, log=log)
    return res
