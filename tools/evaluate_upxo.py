"""Reproducible UPXO evaluation; this is not a production meshing backend.

Run prepare/verify with FESTIM's Python and mesh/worker with UPXO's Python.
See docs/upxo_evaluation.md for environment setup and acceptance criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples/results/upxo-evaluation"
METHODS = (
    "raw",
    "taubin",
    "moving_average",
    "mean_coords",
    "automatic_seeds",
    "reconstruction",
    "taubin_gentle",
)
DEFAULT_METHODS = METHODS[:5]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synthetic_cases():
    """Grains with holes, multiple components, narrow features and junctions."""
    island = np.ones((24, 24), dtype=np.int32)
    island[8:16, 8:16] = 2
    nested = np.ones_like(island)
    nested[4:20, 4:20] = 2
    nested[9:15, 9:15] = 3
    twins = np.ones_like(island)
    for row in range(24):
        col = 6 + row // 3
        twins[row, col:] = 2
        twins[row, col : col + 1] = 3
        twins[row, 19:21] = 4
    junction = np.ones_like(island)
    junction[:12, 12:] = 2
    junction[12:] = 3
    disconnected = np.ones_like(island)
    disconnected[5:9, 5:9] = 2
    disconnected[15:19, 15:19] = 2
    pinch = np.ones_like(island)
    pinch[6:12, 6:12] = 2
    pinch[12:18, 12:18] = 2
    void = island.copy()
    void[8:16, 8:16] = 0
    return {
        "island": island,
        "nested_islands": nested,
        "thin_twins": twins,
        "triple_junction": junction,
        "disconnected_grain": disconnected,
        "diagonal_contact": pinch,
        "interior_void": void,
    }


def prepare(output):
    from festim_microstructure.ebsd.convert import CtfConversion
    from festim_microstructure.ebsd.orientation import (
        euler_bunge_to_quat,
        quat_to_rodrigues,
    )
    from festim_microstructure.ebsd.settings import Settings

    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, labels in synthetic_cases().items():
        gids = np.arange(1, int(labels.max()) + 1)
        ori = quat_to_rodrigues(euler_bunge_to_quat(17 * gids, 23 * gids, 11 * gids))
        # Deliberately anisotropic, to expose implicit square-pixel assumptions.
        np.savez(inputs / f"{name}.npz", labels=labels, vox=[1.6, 0.8], ori=ori)
    ctf = ROOT / "examples/data/D7 PBF SS316L.ctf"
    options = Settings(
        ctf=str(ctf),
        crop="0,306,0,306",
        min_pixels=15,
        max_mad=1.5,
        allow_error=True,
        diagnostics=False,
    )
    conversion = CtfConversion(options).read().import_grains().measure()
    conversion.write(inputs / "ctf_map-ebsd.npz")
    np.savez(
        inputs / "ctf_map.npz",
        labels=conversion.cellids,
        vox=conversion.vox,
        ori=quat_to_rodrigues(conversion.qcell),
    )
    save_json(
        output / "input_provenance.json",
        {
            "ctf_sha256": hashlib.sha256(ctf.read_bytes()).hexdigest(),
            "ctf_shape": list(conversion.cellids.shape),
            "ctf_grains": conversion.ncells,
            "ctf_rms_deg": conversion.segmentation.indexed.rms,
            "importer": "UPXO/DefDAP",
            "crop": options.crop,
            "min_pixels": options.min_pixels,
            "max_mad": options.max_mad,
            "allow_error": options.allow_error,
            "threshold_deg": options.threshold,
        },
    )


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


def geometry_metrics(reference, cells):
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
        "lost_neighbor_pairs": sorted(expected_pairs - actual_pairs),
        "added_neighbor_pairs": sorted(actual_pairs - expected_pairs),
        "accepted": bool(
            not missing
            and not extra
            and all(p.is_valid for p in cells.values())
            and max(gap, excess, overlap) < 1e-7
            and max_area <= 0.05
            and max_boundary_displacement <= 1.0 + 1e-7
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
        "accepted": bool(
            np.all(area > 0)
            and not nonmanifold
            and not duplicate_nodes
            and not unused_curve_segments
            and not missing_curve_segments
            and set(map(int, owners)) == set(cells)
            and max_area < 1e-7
        ),
    }


def adapted_export(mesher, base, vox, ori, labels):
    """Evaluation-only export of the unchanged UPXO triangles with FESTIM tags."""
    import gmsh
    import meshio

    valid = np.flatnonzero(np.isfinite(mesher.nodes[:, 0]))
    index = np.full(len(mesher.nodes), -1, dtype=int)
    index[valid] = np.arange(len(valid))
    points = mesher.nodes[valid].copy()
    points[:, :2] *= np.asarray(vox) * 1e-6
    triangles = index[mesher.elConn["triangle"]]
    owners = mesher._elem_owners["triangle"]
    arcs, _ = boundary_arcs(triangles, owners)
    lines, edge_ids, pairs = [], [], []
    for edge_id, (pair, edges) in enumerate(arcs, start=1):
        lines.extend(edges)
        edge_ids.extend([edge_id] * len(edges))
        pairs.append(pair)
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
    ny, nx = labels.shape
    save_json(
        Path(f"{base}-metadata.json"),
        {"extent_m": [nx * vox[0] * 1e-6, ny * vox[1] * 1e-6], "unit": 1e-6},
    )
    q = np.column_stack((np.ones(len(ori)), ori))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pixel_q = np.zeros((*labels.shape, 4))
    pixel_q[..., 0] = 1
    np.savez_compressed(
        f"{base}-ebsd.npz",
        version=np.array(1),
        labels=labels,
        grain_quats=q,
        vox=vox,
        pixel_quats=pixel_q,
        indexed=labels > 0,
        source_labels=labels,
        crysym=np.array("cubic"),
    )
    np.savez(
        str(base) + "-expected.npz",
        pairs=pairs,
        grain_ids=np.unique(owners),
        ori=ori,
        points=points,
        triangles=triangles,
        owners=owners,
        lines=np.asarray(lines),
        edge_ids=edge_ids,
    )


def worker(input_path, method, directory):
    import matplotlib as mpl

    mpl.use("Agg")
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH
    from upxo.meshing.gsmesh2d import _flatten_cells

    directory.mkdir(parents=True, exist_ok=True)
    data = np.load(input_path)
    labels, vox, ori = data["labels"], data["vox"], data["ori"]
    result = {"case": input_path.stem, "method": method}
    started = time.monotonic()
    try:
        reference = raster_polygons(labels)
        if method == "raw":
            cells = reference
        else:
            from upxo.pxtalops.gssmooth2d import smooth_gs_slice

            # Interior pixel-centre seeds avoid the default generator's outside
            # guard seeds. Stay just below half-integers so UPXO's nearest
            # array-index sampling selects the pixel containing each seed.
            seeds = pixel_seeds(labels.shape)
            smooth = smooth_gs_slice(
                labels,
                seeds=None if method == "automatic_seeds" else seeds,
                area_threshold=0,
                smooth_iter=0 if method == "reconstruction" else 5,
                smooth_lambda=0.25 if method == "taubin_gentle" else 0.5,
                smooth_mu=-0.265 if method == "taubin_gentle" else -0.53,
                method="taubin"
                if method in ("automatic_seeds", "reconstruction", "taubin_gentle")
                else method,
                trim_bounds=(0, 0, labels.shape[1], labels.shape[0]),
                fix_diagonal=False,
                merge_enclosed=False,
                jitter_factor=0,
            )
            # Apply the returned mapping in reverse; do not assume row IDs survived.
            inverse = {new: old for old, new in smooth["old_to_new_gid"].items()}
            cells = {
                inverse[gid]: polygon
                for gid, polygon in smooth["cells"].items()
                if inverse[gid] > 0
            }
        result["geometry"] = geometry_metrics(reference, cells)
        save_json(directory / "result.json", result)
        flat, gid_map = _flatten_cells(cells)
        m = confMesh2dGMSH()
        m.femesh_gmsh(
            flat,
            gid_map=gid_map,
            mesh_size_gb=0.75,
            mesh_size_bulk=4.5,
            mesh_algo=6,
            mesh_order=1,
            recombine_to_quads=False,
            n_threads=1,
            out_dir=str(directory),
            basename="native",
            formats=["msh"],
        )
        result["mesh"] = mesh_metrics(m, cells)
        # Keep UPXO's geometry and triangles untouched. Export scaling uses both axes.
        adapted_export(m, directory / "adapted", vox, ori, labels)
        result["accepted_before_import"] = (
            result["geometry"]["accepted"] and result["mesh"]["accepted"]
        )
        if input_path.stem == "ctf_map":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(
                labels,
                origin="lower",
                extent=(0, labels.shape[1], 0, labels.shape[0]),
                interpolation="nearest",
                cmap="tab20",
                alpha=0.45,
            )
            for geom in cells.values():
                for part in polygon_parts(geom):
                    for ring in [part.exterior, *part.interiors]:
                        ax.plot(*ring.xy, color="black", linewidth=0.35)
            ax.set(
                title=f"CTF pixels and UPXO polygons: {method}",
                xlabel="x / pixels",
                ylabel="y / pixels",
            )
            fig.savefig(directory / "overlay.png", dpi=160)
            plt.close(fig)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["accepted_before_import"] = False
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    save_json(directory / "result.json", result)
    status = "PASS" if result.get("accepted_before_import") else "FAIL"
    print(f"{input_path.stem}/{method}: {status}", flush=True)


def mesh_all(output, cases, methods, jobs, timeout):
    packages = (
        "upxo",
        "numpy",
        "scipy",
        "shapely",
        "gmsh",
        "rasterio",
        "pyvista",
        "meshio",
    )
    direct_url = importlib.metadata.distribution("upxo").read_text("direct_url.json")
    source = json.loads(direct_url) if direct_url else {}
    revision = source.get("vcs_info", {}).get("commit_id")
    if source.get("url", "").startswith("file://"):
        checkout = Path(source["url"].removeprefix("file://"))
        if (checkout / ".git").exists():
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
    observed_methods = sorted(
        {p.parent.name for p in output.glob("*/*/result.json")} | set(methods)
    )
    save_json(
        output / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "upxo_commit": revision,
            "methods": observed_methods,
            "smoothing_iterations": {
                m: 0 if m == "reconstruction" else 5
                for m in observed_methods
                if m != "raw"
            },
            "taubin_weights": {
                "taubin": [0.5, -0.53],
                "taubin_gentle": [0.25, -0.265],
            },
            "mesh_size_gb_px": 0.75,
            "mesh_size_bulk_px": 4.5,
            "mesh_algorithm": 6,
            "threads_per_worker": 1,
            "timeout_s": timeout,
        },
    )
    tasks = [
        (p, method)
        for p in sorted(
            p for p in (output / "inputs").glob("*.npz") if not p.stem.endswith("-ebsd")
        )
        if not cases or p.stem in cases
        for method in methods
    ]

    def run(task):
        path, method = task
        directory = output / path.stem / method
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--input",
            str(path),
            "--method",
            method,
            "--output",
            str(directory),
        ]
        env = dict(
            os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
        )
        with (directory / "run.log").open("w") as log:
            try:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=timeout,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"worker exit {completed.returncode}; see run.log"
                    )
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                result_path = directory / "result.json"
                result = (
                    json.loads(result_path.read_text()) if result_path.exists() else {}
                )
                result.update(
                    case=path.stem,
                    method=method,
                    error=str(exc),
                    accepted_before_import=False,
                )
                save_json(result_path, result)
        result = json.loads((directory / "result.json").read_text())
        status = "PASS" if result.get("accepted_before_import") else "FAIL"
        print(f"{path.stem}/{method}: {result.get('error') or status}", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, tasks))


def verify(output):
    from mpi4py import MPI

    from dolfinx.io import gmsh as gmshio

    from festim_microstructure.formats.msh4 import read_mesh
    from festim_microstructure.meshing.ebsd import EbsdMicrostructure, read_extent

    results = []
    for path in sorted(output.glob("*/*/result.json")):
        result = json.loads(path.read_text())
        directory = path.parent
        native = directory / "native.msh"
        if native.exists():
            try:
                loaded = gmshio.read_from_msh(str(native), MPI.COMM_WORLD, 0, gdim=2)
                result["native_import"] = {
                    "cell_ids": np.unique(loaded.cell_tags.values).tolist(),
                    "facet_tags": loaded.facet_tags is not None
                    and loaded.facet_tags.values.size > 0,
                }
            except Exception as exc:
                result["native_import"] = {"error": f"{type(exc).__name__}: {exc}"}
        base = directory / "adapted"
        if base.with_suffix(".msh4").exists():
            try:
                expected = np.load(str(base) + "-expected.npz")
                mesh, cells, facets = read_mesh(base, gdim=2)
                # EbsdMicrostructure's vertex midpoint query needs this in
                # DOLFINx; a direct Gmsh import does not create it.
                mesh.topology.create_connectivity(0, mesh.topology.dim)
                extent = read_extent(base)
                micro = EbsdMicrostructure.from_mesh(base, mesh, cells, facets, extent)
                raw = output / result["case"] / "raw/adapted-expected.npz"
                if raw.exists():
                    raw_pairs = np.load(raw)["pairs"]
                    before = {tuple(map(int, p)) for p in raw_pairs if 0 not in p}
                    after = {
                        tuple(map(int, p)) for p in expected["pairs"] if 0 not in p
                    }
                    result["final_mesh_neighbors"] = {
                        "lost_pairs": sorted(before - after),
                        "added_pairs": sorted(after - before),
                    }
                np.testing.assert_array_equal(
                    np.unique(cells.values), expected["grain_ids"]
                )
                np.testing.assert_array_equal(
                    np.unique(facets.values), np.arange(1, len(expected["pairs"]) + 1)
                )
                np.testing.assert_array_equal(
                    np.column_stack([micro.edges["grain_a"], micro.edges["grain_b"]]),
                    expected["pairs"],
                )
                np.testing.assert_allclose(
                    mesh.geometry.x.max(axis=0)[:2], extent, rtol=1e-12, atol=1e-15
                )
                from festim_microstructure.ebsd.orientation import (
                    rodrigues_to_quat,
                    cubic_disorientation_angle,
                    qmul,
                    qconj,
                )

                error = cubic_disorientation_angle(
                    qmul(
                        qconj(rodrigues_to_quat(expected["ori"])),
                        rodrigues_to_quat(micro.ori),
                    )
                )
                assert error.max() < 1e-4
                from dolfinx.mesh import compute_midpoints
                from scipy.spatial import cKDTree

                centroids = expected["points"][expected["triangles"]].mean(axis=1)
                cmap = mesh.topology.index_map(mesh.topology.dim)
                local_cells = np.arange(cmap.size_local, dtype=np.int32)
                midpoints = compute_midpoints(mesh, mesh.topology.dim, local_cells)
                distance, rows = cKDTree(centroids).query(midpoints)
                np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-14)
                local_tags = np.zeros(cmap.size_local + cmap.num_ghosts, dtype=int)
                local_tags[cells.indices] = cells.values
                np.testing.assert_array_equal(
                    local_tags[local_cells], expected["owners"][rows]
                )
                # Independently check the boundary network lengths in SI units.
                xyz, line, tags = (
                    expected["points"],
                    expected["lines"],
                    expected["edge_ids"],
                )
                lengths = np.linalg.norm(xyz[line[:, 1]] - xyz[line[:, 0]], axis=1)
                edge_lengths = np.bincount(tags, weights=lengths)[1:]
                np.testing.assert_allclose(
                    micro.edges["length"], edge_lengths, rtol=1e-10, atol=1e-15
                )
                if result["case"] != "interior_void":
                    micro.check_orientations()
                result["adapted_import"] = {
                    "accepted": True,
                    "grains": micro.n_grains,
                    "boundary_arcs": micro.edges.n,
                    "triple_junctions": micro.triple_junctions,
                    "network_length_m": micro.network_measure,
                    "facet_tags": True,
                    "orientation_ids_verified": True,
                    "extent_m": list(extent),
                }
            except Exception as exc:
                result["adapted_import"] = {
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        result["accepted"] = bool(
            result.get("accepted_before_import")
            and result.get("adapted_import", {}).get("accepted")
        )
        save_json(path, result)
        results.append(result)
        status = "PASS" if result["accepted"] else "FAIL"
        print(f"{result['case']}/{result['method']}: {status}", flush=True)
    save_json(output / "summary.json", results)
    save_json(
        output / "solver_environment.json",
        {
            "python": sys.version,
            "dolfinx": importlib.metadata.version("fenics-dolfinx"),
            "numpy": np.__version__,
            "mpi_size": MPI.COMM_WORLD.size,
        },
    )
    rows = [
        "| Case | Method | Geometry | Mesh | Native facet tags | "
        "Adapted import | Overall |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:

        def yes(value):
            return "PASS" if value else "FAIL"

        values = [
            r["case"],
            r["method"],
            yes(r.get("geometry", {}).get("accepted")),
            yes(r.get("mesh", {}).get("accepted")),
            str(bool(r.get("native_import", {}).get("facet_tags"))),
            yes(r.get("adapted_import", {}).get("accepted")),
            yes(r["accepted"]),
        ]
        rows.append("| " + " | ".join(values) + " |")
    (output / "summary.md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "mesh", "worker", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--method", choices=METHODS, default="raw")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "worker":
        worker(args.input, args.method, args.output)
    elif args.stage == "mesh":
        mesh_all(args.output, args.cases, args.methods, args.jobs, args.timeout)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
