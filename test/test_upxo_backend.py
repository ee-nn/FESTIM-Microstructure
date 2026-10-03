"""Geometry preservation, failure handling and UPXO interpreter isolation."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from festim_microstructure.ebsd.orientation import rodrigues_to_quat
from festim_microstructure.formats.ebsd import EbsdData, read_ebsd, write_ebsd
from festim_microstructure.meshing import upxo


def test_configured_interpreter_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("FM_UPXO_PYTHON", "/does/not/exist/python")
    assert upxo.resolve_python(required=False) is None
    with pytest.raises(FileNotFoundError, match="FM_UPXO_PYTHON"):
        upxo.resolve_python()


@pytest.mark.parametrize(
    "field,value",
    [
        ("smoothing", "moving_average"),
        ("smooth_iter", -1),
        ("n_threads", 0),
        ("mesh_size_gb", float("nan")),
        ("smooth_mu", 0),
        ("timeout", 0),
    ],
)
def test_reject_unsupported_or_invalid_options(field, value):
    with pytest.raises(ValueError):
        upxo.UpxoMeshOptions(**{field: value})


def test_worker_failure_preserves_accepted_outputs(tmp_path, monkeypatch):
    labels = np.ones((4, 4), dtype=np.int32)
    source = tmp_path / "map.npz"
    write_ebsd(source, EbsdData(labels, np.array([[1.0, 0, 0, 0]]), (1, 1)))
    mesh = tmp_path / "poly.msh4"
    mesh.write_bytes(b"previous accepted mesh")
    monkeypatch.setattr(upxo, "resolve_python", lambda explicit: "/fake/python")

    def failed_worker(command, **kwargs):
        stage = Path(command[-1]).parent
        (stage / "poly-validation.json").write_text(
            json.dumps(
                {
                    "accepted": False,
                    "error": "geometry changed",
                }
            )
        )
        (stage / "poly.msh4").write_bytes(b"rejected mesh")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(upxo.subprocess, "run", failed_worker)
    with pytest.raises(RuntimeError, match="UPXO meshing failed"):
        upxo.mesh_ebsd(source, workdir=tmp_path)
    assert mesh.read_bytes() == b"previous accepted mesh"
    assert (
        json.loads((tmp_path / "poly-failed-validation.json").read_text())["accepted"]
        is False
    )
    assert not list(tmp_path.glob(".poly-upxo-*"))


def test_geometry_gate_rejects_lost_grain_contact():
    pytest.importorskip("shapely")
    from shapely.geometry import box

    reference = {1: box(0, 0, 1, 1), 2: box(1, 0, 2, 1)}
    moved = {1: reference[1], 2: box(1.01, 0, 2, 1)}
    report = upxo.geometry_metrics(reference, moved)
    assert not report["accepted"]
    assert report["lost_neighbor_pairs"] == [(1, 2)]
    assert report["gap_fraction"] > 0


@pytest.mark.parametrize("smoothing", ["none", "taubin"])
def test_real_worker_retains_island_and_ownership(tmp_path, smoothing):
    if upxo.resolve_python(required=False) is None:
        pytest.skip("configure FM_UPXO_PYTHON for the UPXO integration test")
    from festim_microstructure.formats.msh4 import read_msh4

    labels = np.ones((24, 24), dtype=np.int32)
    labels[8:16, 8:16] = 2
    source = tmp_path / "map.npz"
    ori = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    write_ebsd(source, EbsdData(labels, rodrigues_to_quat(ori), (1.6, 0.8)))
    base = upxo.mesh_ebsd(
        source, upxo.UpxoMeshOptions(smoothing=smoothing), workdir=tmp_path
    )
    report = json.loads(Path(f"{base}-validation.json").read_text())
    assert report["accepted"]
    assert report["geometry"]["actual_topology"] == {
        "1": {"parts": 1, "holes": 1},
        "2": {"parts": 1, "holes": 0},
    }
    xyz, lines, triangles = read_msh4(base.with_suffix(".msh4"))
    assert {g for g, _ in triangles} == {1, 2}
    assert {tag for tag, _ in lines} == {1, 2}
    np.testing.assert_allclose(
        np.max(list(xyz.values()), axis=0)[:2], [38.4e-6, 19.2e-6]
    )
    np.testing.assert_allclose(
        read_ebsd(f"{base}-ebsd.npz").grain_quats, rodrigues_to_quat(ori)
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("island", {"1": {"parts": 1, "holes": 1}, "2": {"parts": 1, "holes": 0}}),
        (
            "disconnected",
            {"1": {"parts": 1, "holes": 2}, "2": {"parts": 2, "holes": 0}},
        ),
        ("void", {"1": {"parts": 1, "holes": 1}}),
    ],
)
def test_triangle_topology_counts_islands_voids_and_disconnected_parts(case, expected):
    from test_upxo_evaluation import pixel_triangles

    labels = np.ones((24, 24), dtype=np.int32)
    if case == "disconnected":
        labels[5:9, 5:9] = labels[15:19, 15:19] = 2
    else:
        labels[8:16, 8:16] = 0 if case == "void" else 2
    triangles, owners = pixel_triangles(labels)
    assert upxo.mesh_topology(triangles, owners) == expected
