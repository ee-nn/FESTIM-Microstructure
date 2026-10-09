"""Validated UPXO EBSD meshing, isolated from the FESTIM interpreter.

The worker is executable as a file under Python 3.13 without installing FESTIM.
Only NumPy is imported until geometry reconstruction or meshing is requested.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

__all__ = [
    "UpxoMeshOptions",
    "import_ebsd",
    "mesh_ebsd",
    "probe_worker",
    "resolve_python",
]
UPXO_VERSION = "1.3.1"
DEFDAP_VERSION = "0.93.6"
UPXO_REVISION = "a53885ef0a7a06f6b3fb195747363ae0a19bb127"
_PROTOCOL = 2


@dataclass
class UpxoMeshOptions:
    """UPXO mesh sizes and controlled smoothing, measured in pixel coordinates.

    ``smoothing='none'`` preserves the raster boundaries exactly. Taubin uses
    explicit pixel seeds and disables grain merging and diagonal repairs.
    ``thin_grain_px`` enables optional thin-grain protection without changing
    pixel labels. For unfilled voids, use ``smoothing="none"``.
    Geometry that fails the preservation checks raises before publication.
    """

    smoothing: str = "taubin"
    smooth_iter: int = 5
    smooth_lambda: float = 0.25
    smooth_mu: float = -0.265
    thin_grain_px: float = 0.0
    mesh_size_gb: float = 0.75
    mesh_size_bulk: float = 4.5
    n_threads: int = 1
    max_area_change: float = 0.05
    max_boundary_displacement: float = 1.0
    python: str | None = None
    timeout: float = 600.0

    def __post_init__(self):
        if self.smoothing not in ("none", "taubin"):
            raise ValueError("smoothing must be 'none' or 'taubin'")
        for name in ("smooth_iter", "n_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.smooth_iter < 0 or self.n_threads < 1:
            raise ValueError("smooth_iter must be >= 0 and n_threads must be >= 1")
        for name in ("mesh_size_gb", "mesh_size_bulk", "timeout"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_area_change", "max_boundary_displacement", "thin_grain_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.smooth_lambda < 1 or not -1 < self.smooth_mu < 0:
            raise ValueError(
                "Taubin requires 0 < smooth_lambda < 1 and -1 < smooth_mu < 0"
            )


def resolve_python(explicit=None, required=True):
    """Choose the UPXO interpreter: explicit path, FM_UPXO_PYTHON, then local.

    The named conda environment beside the current one is also discovered.
    An explicitly configured missing interpreter is an error, not a fallback.
    """
    configured = explicit or os.environ.get("FM_UPXO_PYTHON")
    if configured:
        executable = shutil.which(str(configured))
        if executable:
            return executable
    else:
        if sys.version_info >= (3, 13) and importlib.util.find_spec("upxo"):
            return sys.executable
        sibling = (
            Path(sys.prefix).parent / "festim-microstructure-upxo" / "bin" / "python"
        )
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)
    if required:
        raise FileNotFoundError(
            "UPXO Python interpreter not found. Create and activate "
            "environment-festim-microstructure-upxo.yml, or set FM_UPXO_PYTHON "
            "to an interpreter with UPXO installed."
        )
    return None


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probe_worker(executable=None, *, health=False, timeout=30.0):
    """Observe the selected worker before cache reuse, or check its EBSD APIs.

    Run in isolation just like the importer/mesher. No probe is cached, so an
    environment replacement at the same interpreter path invalidates results.
    """
    try:
        executable = resolve_python(executable)
    except FileNotFoundError as exc:
        raise RuntimeError(f"UPXO environment probe failed: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="upxo-probe-") as scratch:
        output = Path(scratch) / "environment.json"
        try:
            result = subprocess.run(
                [
                    executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--health" if health else "--environment",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise RuntimeError(f"UPXO environment probe failed: {detail}") from exc
        try:
            return json.loads(output.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"UPXO environment probe returned no valid report: {result.stdout}"
            ) from exc


def _environment_report(*, health=False):
    """Fingerprint installed distribution metadata and test EBSD runtime imports."""
    if sys.version_info < (3, 13):
        raise RuntimeError("UPXO requires Python 3.13 or newer")
    _check_compatibility(require_defdap=True)
    required = (
        "upxo",
        "defdap",
        "numpy",
        "scipy",
        "shapely",
        "gmsh",
        "rasterio",
        "meshio",
        "pyvista",
    )
    packages = {name: importlib.metadata.version(name) for name in required}
    distributions = []
    for dist in importlib.metadata.distributions():
        distributions.append(
            {
                "name": dist.metadata["Name"],
                "version": dist.version,
                "origin": dist.read_text("direct_url.json"),
                "record_sha256": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
        )
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": packages,
        "distributions": sorted(distributions, key=lambda d: (d["name"], d["version"])),
    }
    if health:
        apis = {
            "upxo.interfaces.defdap.ebsd_reader": ("EBSDReader",),
            "upxo.meshing.conformal_mesher2d": ("confMesh2dGMSH",),
            "upxo.meshing.gsmesh2d": ("_flatten_cells",),
            "upxo.pxtalops.gssmooth2d": ("smooth_gs_slice",),
            "rasterio.features": ("shapes",),
            "shapely.geometry": ("Polygon",),
            "meshio": ("write",),
            "gmsh": ("initialize", "finalize"),
        }
        for module, attributes in apis.items():
            imported = importlib.import_module(module)
            for attribute in attributes:
                getattr(imported, attribute)
        report["health"] = "ok"
    return report


def mesh_ebsd(
    archive, options=None, *, workdir="results", stem="poly", unit=1e-6, force=True
):
    """Mesh a native EBSD archive and write tagged SI and diagnostic outputs.

    Cache reuse requires matching input bytes, options, units, interpreter and
    observed installed distributions and worker code, plus output checksums.
    Failed runs leave previously
    accepted outputs intact and keep a log and validation report for diagnosis.
    """
    from festim_microstructure.formats.ebsd import read_ebsd

    options = options or UpxoMeshOptions()
    options.__post_init__()
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("unit must be finite and positive")
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("stem must be a filename without directory components")
    archive = Path(archive).resolve()
    data = read_ebsd(archive)
    labels, vox, ori = data.labels, np.asarray(data.vox), data.grain_quats
    executable = resolve_python(options.python)
    base = Path(workdir).resolve() / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    settings = asdict(options)
    settings["python"] = executable
    identity = {
        "protocol": _PROTOCOL,
        "upxo_revision": UPXO_REVISION,
        "worker_environment": probe_worker(executable, timeout=options.timeout),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "options": settings,
        "unit": unit,
    }
    manifest = Path(f"{base}-upxo.json")
    if not force and manifest.is_file():
        try:
            saved = json.loads(manifest.read_text())
            outputs = saved["outputs"]
            if (
                saved["identity"] == identity
                and outputs
                and all(
                    (base.parent / name).is_file()
                    and hashlib.sha256((base.parent / name).read_bytes()).hexdigest()
                    == digest
                    for name, digest in outputs.items()
                )
            ):
                return base
        except (ValueError, KeyError, OSError):
            pass
    with tempfile.TemporaryDirectory(
        prefix=f".{stem}-upxo-", dir=base.parent
    ) as scratch:
        stage = Path(scratch)
        np.savez(stage / "input.npz", labels=labels.astype(np.int32), vox=vox, ori=ori)
        request = stage / "request.json"
        _save_json(request, {"options": settings, "unit": unit, "stem": stem})
        log = Path(f"{base}-upxo.log")
        with log.open("w") as stream:
            try:
                subprocess.run(
                    [
                        executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(request),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=options.timeout,
                    check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                failed = stage / f"{stem}-validation.json"
                reason = "worker exited before validation"
                if failed.is_file():
                    shutil.copyfile(failed, f"{base}-failed-validation.json")
                    reason = json.loads(failed.read_text()).get("error", reason)
                if isinstance(exc, subprocess.TimeoutExpired):
                    reason = f"exceeded {options.timeout:g} seconds"
                raise RuntimeError(f"UPXO meshing failed: {reason}; see {log}") from exc
        shutil.copyfile(archive, stage / f"{stem}-ebsd.npz")
        outputs = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in stage.iterdir()
            if p.name.startswith(stem) and p.is_file()
        }
        for name in outputs:
            (stage / name).replace(base.parent / name)
        _save_json(manifest, {"identity": identity, "outputs": outputs})
    return base


def raster_polygons(labels):
    """Polygonize exact pixels, retaining holes and shared edge subdivisions.

    Rasterio may simplify a straight edge differently on opposite sides of
    a junction. Restore each integer grid vertex so UPXO receives identical
    segment endpoints on both sides. No pixel labels or topology are repaired.
    """
    from rasterio.features import shapes
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    def dense_ring(ring):
        result = []
        coords = np.asarray(ring.coords)
        for a, b in pairwise(coords):
            n = round(np.abs(b - a).max())
            result.extend(a + (b - a) * k / n for k in range(n))
        return result

    grouped = defaultdict(list)
    for geometry, grain in shapes(labels, mask=labels > 0, connectivity=4):
        poly = shape(geometry)
        grouped[int(grain)].append(
            Polygon(
                dense_ring(poly.exterior),
                [dense_ring(r) for r in poly.interiors],
            )
        )
    return {gid: unary_union(parts) for gid, parts in grouped.items()}


def polygon_parts(geometry):
    return [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)


def pixel_seeds(shape):
    """Seeds inside each pixel that UPXO's nearest-index lookup samples correctly."""
    yy, xx = np.indices(shape, dtype=float)
    # nextafter(0.5, 0) is too close: ndimage's nearest rounding still picks 1.
    return np.column_stack([xx.ravel(), yy.ravel()]) + 0.5 - 1e-6


def geometry_metrics(reference, cells, max_area_change=0.05, max_displacement=1.0):
    """Check the partition against the original pixels, not just the mesh input."""
    from shapely.ops import unary_union

    expected = unary_union(list(reference.values()))
    actual = unary_union(list(cells.values()))
    scale = expected.area
    missing = sorted(set(reference) - set(cells))
    extra = sorted(set(cells) - set(reference))
    area_errors, displaced = {}, {}
    for gid in set(reference) & set(cells):
        original, new = reference[gid], cells[gid]
        area_errors[gid] = abs(new.area - original.area) / original.area
        displaced[gid] = original.symmetric_difference(new).area

    def topology(polygons):
        return {
            str(gid): {
                "parts": len(polygon_parts(geom)),
                "holes": sum(len(part.interiors) for part in polygon_parts(geom)),
            }
            for gid, geom in polygons.items()
        }

    def neighbors(polygons):
        gids = sorted(polygons)
        return {
            (a, b)
            for i, a in enumerate(gids)
            for b in gids[i + 1 :]
            if polygons[a].boundary.intersection(polygons[b].boundary).length > 1e-7
        }

    expected_pairs, actual_pairs = neighbors(reference), neighbors(cells)
    gap = expected.difference(actual).area / scale
    excess = actual.difference(expected).area / scale
    overlap = max(0.0, sum(g.area for g in cells.values()) - actual.area) / scale
    topo_reference, topo_actual = topology(reference), topology(cells)
    max_area = max(area_errors.values(), default=0.0)
    max_boundary_displacement = max(
        (
            reference[g].boundary.hausdorff_distance(cells[g].boundary)
            for g in set(reference) & set(cells)
        ),
        default=0.0,
    )
    return {
        "missing_ids": missing,
        "extra_ids": extra,
        "invalid_ids": sorted(g for g, p in cells.items() if not p.is_valid),
        "gap_fraction": gap,
        "excess_fraction": excess,
        "overlap_fraction": overlap,
        "max_grain_area_relative_error": max_area,
        "mean_grain_area_relative_error": float(np.mean(list(area_errors.values()))),
        "label_displacement_fraction": sum(displaced.values()) / (2 * scale),
        "max_boundary_displacement_px": max_boundary_displacement,
        "reference_topology": topo_reference,
        "actual_topology": topo_actual,
        "neighbor_pairs": sorted(expected_pairs),
        "lost_neighbor_pairs": sorted(expected_pairs - actual_pairs),
        "added_neighbor_pairs": sorted(actual_pairs - expected_pairs),
        "accepted": bool(
            not missing
            and not extra
            and all(p.is_valid for p in cells.values())
            and max(gap, excess, overlap) < 1e-7
            and max_area <= max_area_change
            and max_boundary_displacement <= max_displacement + 1e-7
            and topo_reference == topo_actual
            and expected_pairs == actual_pairs
        ),
    }


def boundary_arcs(triangles, owners):
    """Group mesh boundary facets into arcs ending at network junctions.

    Closed loops need no junction. Separate loops between the same grain pair
    get separate IDs; bulk edges inside one grain receive no boundary tag.
    """
    adjacency = defaultdict(list)
    for triangle, grain in zip(triangles, owners, strict=True):
        for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
            adjacency[tuple(sorted((int(a), int(b))))].append(int(grain))
    nonmanifold = sum(len(grains) > 2 for grains in adjacency.values())
    boundary = {
        edge: tuple(sorted(grains)) if len(grains) == 2 else (grains[0], 0)
        for edge, grains in adjacency.items()
        if len(grains) == 1 or (len(grains) == 2 and grains[0] != grains[1])
    }
    incidence = defaultdict(list)
    for edge in boundary:
        for node in edge:
            incidence[node].append(edge)
    unseen = set(boundary)
    arcs = []
    while unseen:
        seed = min(unseen)
        stack, edges = [seed], []
        unseen.remove(seed)
        while stack:
            edge = stack.pop()
            edges.append(edge)
            for node in edge:
                if len(incidence[node]) == 2:
                    for other in incidence[node]:
                        if other in unseen and boundary[other] == boundary[seed]:
                            unseen.remove(other)
                            stack.append(other)
        arcs.append((boundary[seed], sorted(edges)))
    return arcs, nonmanifold


def mesh_topology(triangles, owners):
    """Count components and holes in each grain's actual triangular submesh.

    Components connect across shared facets. Euler counts use separate vertices
    for separate components, preserving corner-only contacts between parts.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(triangles)
    edges = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    triangle_ids = np.tile(np.arange(n), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges, ordered_ids = edges[order], triangle_ids[order]
    shared = np.flatnonzero(np.all(ordered_edges[1:] == ordered_edges[:-1], axis=1))
    a, b = ordered_ids[shared], ordered_ids[shared + 1]
    same = owners[a] == owners[b]
    graph = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n))
    n_parts, component = connected_components(graph, directed=False)
    vertices = np.unique(
        np.column_stack([np.repeat(component, 3), triangles.ravel()]), axis=0
    )
    component_edges = np.unique(
        np.column_stack([component[triangle_ids], edges]), axis=0
    )
    v = np.bincount(vertices[:, 0], minlength=n_parts)
    e = np.bincount(component_edges[:, 0], minlength=n_parts)
    f = np.bincount(component, minlength=n_parts)
    holes = 1 - (v - e + f)
    part_owner = np.zeros(n_parts, dtype=owners.dtype)
    part_owner[component] = owners
    return {
        str(g): {
            "parts": int((part_owner == g).sum()),
            "holes": int(holes[part_owner == g].sum()),
        }
        for g in np.unique(owners)
    }


def mesh_metrics(mesher, cells):
    triangles = mesher.elConn["triangle"]
    owners = mesher._elem_owners["triangle"]
    xy = mesher.nodes[triangles, :2]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    area = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    arcs, nonmanifold = boundary_arcs(triangles, owners)
    expected_edges = {e for _, edges in arcs for e in edges}
    actual_edges = {tuple(sorted(map(int, e))) for e in mesher.GBlines}
    unused_curve_segments = len(actual_edges - expected_edges)
    missing_curve_segments = len(expected_edges - actual_edges)
    valid_nodes = np.flatnonzero(np.isfinite(mesher.nodes[:, 0]))
    xy_nodes = mesher.nodes[valid_nodes, :2]
    duplicate_nodes = len(xy_nodes) - len(np.unique(np.round(xy_nodes, 8), axis=0))
    max_area = max(
        abs(area[owners == gid].sum() - geom.area) / geom.area
        for gid, geom in cells.items()
    )
    topology = mesh_topology(triangles, owners)
    expected_topology = {
        str(g): {
            "parts": len(polygon_parts(p)),
            "holes": sum(len(part.interiors) for part in polygon_parts(p)),
        }
        for g, p in cells.items()
    }
    return {
        "nodes": len(valid_nodes),
        "triangles": len(triangles),
        "inverted_or_zero_triangles": int((area <= 0).sum()),
        "nonmanifold_facets": nonmanifold,
        "duplicate_coordinate_nodes": duplicate_nodes,
        "unused_curve_segments": unused_curve_segments,
        "missing_curve_segments": missing_curve_segments,
        "missing_grain_ids": sorted(set(cells) - set(map(int, owners))),
        "max_mesh_polygon_area_relative_error": float(max_area),
        "min_angle_deg": float(
            mesher.quality_report["min_angle_deg"]["triangle"]["min"]
        ),
        "boundary_arcs": len(arcs),
        "actual_topology": topology,
        "accepted": bool(
            np.all(area > 0)
            and not nonmanifold
            and not duplicate_nodes
            and not unused_curve_segments
            and not missing_curve_segments
            and set(map(int, owners)) == set(cells)
            and max_area < 1e-7
            and topology == expected_topology
        ),
    }


def export_mesh(mesher, base, vox, ori, shape, unit):
    """Retain UPXO's triangles and add deterministic grain and boundary tags."""
    import gmsh
    import meshio

    valid = np.flatnonzero(np.isfinite(mesher.nodes[:, 0]))
    index = np.full(len(mesher.nodes), -1, dtype=int)
    index[valid] = np.arange(len(valid))
    points = mesher.nodes[valid].copy()
    points[:, :2] *= np.asarray(vox) * unit
    triangles = index[mesher.elConn["triangle"]]
    owners = mesher._elem_owners["triangle"]
    arcs, _ = boundary_arcs(triangles, owners)
    lines, edge_ids = [], []
    for edge_id, (_pair, edges) in enumerate(arcs, start=1):
        lines.extend(edges)
        edge_ids.extend([edge_id] * len(edges))
    edge_ids = np.asarray(edge_ids)
    field_data = {f"face{g}": np.array([g, 2]) for g in np.unique(owners)}
    field_data.update({f"edge{e}": np.array([e, 1]) for e in np.unique(edge_ids)})
    intermediate = base.with_suffix(".msh")
    meshio.write(
        intermediate,
        meshio.Mesh(
            points,
            [("line", np.asarray(lines)), ("triangle", triangles)],
            cell_data={
                "gmsh:physical": [edge_ids, owners],
                "gmsh:geometrical": [edge_ids, owners],
            },
            field_data=field_data,
        ),
        file_format="gmsh22",
        binary=False,
    )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(intermediate))
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(base.with_suffix(".msh4")))
    finally:
        gmsh.finalize()
    intermediate.unlink()
    ny, nx = shape
    _save_json(
        Path(f"{base}-metadata.json"),
        {
            "version": 1,
            "extent_m": [nx * vox[0] * unit, ny * vox[1] * unit],
            "unit": unit,
            "n_grains": len(ori),
            "crystal_symmetry": "cubic",
        },
    )


def _check_compatibility(*, require_defdap=False):
    """Enforce the same evaluated versions and source revision in both workers."""
    if importlib.metadata.version("upxo") != UPXO_VERSION:
        raise RuntimeError(f"This backend requires evaluated UPXO {UPXO_VERSION}")
    distribution = importlib.metadata.distribution("upxo")
    origin = distribution.read_text("direct_url.json")
    revision = (
        json.loads(origin).get("vcs_info", {}).get("commit_id") if origin else None
    )
    if revision and revision != UPXO_REVISION:
        raise RuntimeError("UPXO is not installed from the evaluated commit")
    if require_defdap and importlib.metadata.version("defdap") != DEFDAP_VERSION:
        raise RuntimeError(f"EBSD import requires DefDAP {DEFDAP_VERSION}")


def _worker(request):
    """Run under the pinned UPXO environment; never import solver libraries."""
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH
    from upxo.meshing.gsmesh2d import _flatten_cells

    request = Path(request)
    config = json.loads(request.read_text())
    options = UpxoMeshOptions(**config["options"])
    directory, stem = request.parent, config["stem"]
    data = np.load(directory / "input.npz")
    labels, vox, ori = data["labels"], data["vox"], data["ori"]
    report = {
        "accepted": False,
        "upxo_version": importlib.metadata.version("upxo"),
        "options": asdict(options),
    }
    distribution = importlib.metadata.distribution("upxo")
    origin = distribution.read_text("direct_url.json")
    source = json.loads(origin) if origin else {}
    revision = source.get("vcs_info", {}).get("commit_id")
    report["upxo_commit"] = revision
    report["packages"] = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "shapely", "gmsh", "rasterio", "meshio")
    }
    validation = directory / f"{stem}-validation.json"
    try:
        _check_compatibility()
        reference = raster_polygons(labels)
        cells = reference
        if options.smoothing == "taubin" and options.smooth_iter:
            from upxo.pxtalops.gssmooth2d import smooth_gs_slice

            smooth = smooth_gs_slice(
                labels,
                seeds=pixel_seeds(labels.shape),
                area_threshold=0,
                smooth_iter=options.smooth_iter,
                smooth_lambda=options.smooth_lambda,
                smooth_mu=options.smooth_mu,
                method="taubin",
                trim_bounds=(0, 0, labels.shape[1], labels.shape[0]),
                fix_diagonal=False,
                merge_enclosed=False,
                jitter_factor=0,
                thin_grain_px=options.thin_grain_px,
                close_staircase=False,
            )
            inverse = {new: old for old, new in smooth["old_to_new_gid"].items()}
            cells = {
                inverse[gid]: poly
                for gid, poly in smooth["cells"].items()
                if inverse[gid] > 0
            }
        report["geometry"] = geometry_metrics(
            reference, cells, options.max_area_change, options.max_boundary_displacement
        )
        _save_json(validation, report)
        if not report["geometry"]["accepted"]:
            raise RuntimeError(
                "UPXO reconstruction/smoothing changed the raster beyond the geometry "
                "limits. Inspect the validation report; reduce smoothing or use "
                "UpxoMeshOptions(smoothing='none')."
            )
        flat, gid_map = _flatten_cells(cells)
        mesher = confMesh2dGMSH()
        mesher.femesh_gmsh(
            flat,
            gid_map=gid_map,
            mesh_size_gb=options.mesh_size_gb,
            mesh_size_bulk=options.mesh_size_bulk,
            mesh_algo=6,
            mesh_order=1,
            recombine_to_quads=False,
            n_threads=options.n_threads,
        )
        report["mesh"] = mesh_metrics(mesher, cells)
        arcs, _ = boundary_arcs(
            mesher.elConn["triangle"], mesher._elem_owners["triangle"]
        )
        actual_pairs = {pair for pair, _ in arcs if 0 not in pair}
        expected_pairs = {tuple(p) for p in report["geometry"]["neighbor_pairs"]}
        report["mesh"]["lost_neighbor_pairs"] = sorted(expected_pairs - actual_pairs)
        report["mesh"]["added_neighbor_pairs"] = sorted(actual_pairs - expected_pairs)
        _save_json(validation, report)
        if not report["mesh"]["accepted"] or actual_pairs != expected_pairs:
            raise RuntimeError(
                "UPXO mesh failed conformity, ownership or adjacency checks"
            )
        export_mesh(mesher, directory / stem, vox, ori, labels.shape, config["unit"])
        report["accepted"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _save_json(validation, report)


def import_ebsd(ctf, window, indexed, settings, log=print):
    """Canonicalize the cropped, filtered CTF and invoke UPXO's EBSDReader.

    DefDAP's text loader reshapes file rows and uses a scalar XStep. Reorder
    by coordinates here, supply absent optional quality columns, and retain
    both pixel spacings in the native archive. Phase/quality rejections become
    phase zero before detection, so they cannot bridge surviving grains.
    """
    executable = resolve_python(settings.python)
    with tempfile.TemporaryDirectory(prefix="upxo-import-") as scratch:
        stage = Path(scratch)
        ny, nx = indexed.shape
        full_ny, full_nx = ctf.shape
        ix = np.rint(ctf["X"] / ctf.header["XStep"]).astype(int)
        iy = np.rint(ctf["Y"] / ctf.header["YStep"]).astype(int)
        inside = (ix >= 0) & (ix < full_nx) & (iy >= 0) & (iy < full_ny)
        columns = [
            "Phase",
            "X",
            "Y",
            "Bands",
            "Error",
            "Euler1",
            "Euler2",
            "Euler3",
            "MAD",
            "BC",
            "BS",
        ]
        arrays = {}
        for name in columns:
            array = np.zeros(ctf.shape)
            if ctf.has(name):
                array[iy[inside], ix[inside]] = ctf[name][inside]
            arrays[name] = array[window].copy()
        arrays["Phase"] = indexed.astype(int)
        # Invalid, rejected orientations do not participate in detection or
        # diagnostics; retain finite rejected measurements for error panels.
        for name in ("Euler1", "Euler2", "Euler3"):
            arrays[name][~np.isfinite(arrays[name])] = 0
        yy, xx = np.indices((ny, nx))
        arrays["X"], arrays["Y"] = xx * ctf.header["XStep"], yy * ctf.header["YStep"]
        lines = ctf.path.read_text().splitlines()
        phase_row = next(
            i for i, line in enumerate(lines) if line.split("\t")[0] == "Phases"
        )
        phase_line = lines[phase_row + settings.phase]
        canonical = stage / "filtered.ctf"
        with canonical.open("w") as stream:
            stream.write(
                f"Channel Text File\nXCells\t{nx}\nYCells\t{ny}\n"
                f"XStep\t{ctf.header['XStep']}\nYStep\t{ctf.header['YStep']}\n"
                f"Phases\t1\n{phase_line}\n"
            )
            stream.write("\t".join(columns) + "\n")
            np.savetxt(
                stream,
                np.column_stack([arrays[col].ravel() for col in columns]),
                fmt="%.12g",
                delimiter="\t",
            )
        request = stage / "request.json"
        _save_json(
            request,
            {"min_pixels": settings.min_pixels, "threshold": settings.threshold},
        )
        logfile = stage / "import.log"
        with logfile.open("w") as stream:
            try:
                subprocess.run(
                    [
                        executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        "--import",
                        str(request),
                    ],
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=settings.timeout,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"UPXO/DefDAP import failed: {logfile.read_text()[-6000:]}"
                ) from exc
        with np.load(stage / "import.npz", allow_pickle=False) as imported:
            result = {
                key: imported[key]
                for key in ("source_labels", "pixel_quats", "grain_quats")
            }
        result["packages"] = json.loads((stage / "packages.json").read_text())
        if log:
            log(
                f"  importer: UPXO {result['packages']['upxo']}, "
                f"DefDAP {result['packages']['defdap']}"
            )
        return result


def _import_worker(request):
    """UPXO import and DefDAP symmetry-aware means, without local segmentation."""
    from upxo.interfaces.defdap.ebsd_reader import EBSDReader

    path = Path(request)
    config = json.loads(path.read_text())
    _check_compatibility(require_defdap=True)
    reader = EBSDReader.load(path.parent / "filtered.ctf")
    try:
        reader.detect_grains(
            min_grain_size=config["min_pixels"], misori_tol=config["threshold"]
        )
    except Exception as exc:
        if str(exc) == "No grains detected.":
            raise ValueError("no grains survived DefDAP grain detection") from exc
        raise
    grains = reader._defdap_map.grainList
    if not grains:
        raise ValueError("no grains survived DefDAP grain detection")
    means = []
    for grain in grains:
        grain.calcAverageOri()
        means.append(grain.refOri.quatCoef)
    labels = np.asarray(reader.lfi_ebsd, dtype=np.int32)
    if not np.array_equal(np.unique(labels[labels > 0]), np.arange(1, len(means) + 1)):
        raise ValueError("UPXO grain IDs differ from DefDAP orientation row order")
    np.savez(
        path.parent / "import.npz",
        source_labels=labels,
        pixel_quats=reader.quat_ebsd,
        grain_quats=np.asarray(means),
    )
    _save_json(
        path.parent / "packages.json",
        {
            p: importlib.metadata.version(p)
            for p in ("upxo", "defdap", "numpy", "scipy")
        },
    )


if __name__ == "__main__":
    modes = {
        "--worker": _worker,
        "--import": _import_worker,
        "--environment": lambda path: _save_json(Path(path), _environment_report()),
        "--health": lambda path: _save_json(
            Path(path), _environment_report(health=True)
        ),
    }
    if len(sys.argv) != 3 or sys.argv[1] not in modes:
        raise SystemExit("usage: upxo.py --worker/--import/--environment/--health PATH")
    modes[sys.argv[1]](sys.argv[2])
