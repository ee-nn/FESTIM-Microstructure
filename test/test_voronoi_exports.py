"""Read actual BP time series to catch lost timesteps or stale field values."""

import importlib.util
from pathlib import Path

import adios2
import numpy as np
import pytest

pytestmark = pytest.mark.fenics


def test_all_3d_voronoi_cases_export_every_timestep(tmp_path, monkeypatch):
    """Both phases retain t=0 and every solved step for all three alignments."""
    script = Path(__file__).resolve().parents[1] / "examples/voronoi_polycrystal_3d.py"
    spec = importlib.util.spec_from_file_location("voronoi_3d_exports", script)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    monkeypatch.setattr(example, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(example, "n_seeds", 6)
    monkeypatch.setattr(example, "h_gb", 0.15)
    monkeypatch.setattr(example, "h_bulk", 0.3)
    monkeypatch.setattr(example, "dt", 0.05)
    monkeypatch.setattr(example, "t_end", 0.15)
    example.main()

    prefixes = ("voronoi3d", "voronoi3d_parallel", "voronoi3d_perpendicular")
    assert len(list(tmp_path.glob("*.bp"))) == 6
    for prefix in prefixes:
        for phase, field in (("grains", "c_lattice"), ("network", "c_gb_1000000")):
            with adios2.FileReader(str(tmp_path / f"{prefix}_{phase}.bp")) as reader:
                assert reader.num_steps() == 4
                times = reader.read("step", step_selection=[0, 4])
                assert np.allclose(times.ravel(), [0.0, 0.05, 0.1, 0.15])
                initial = reader.read(field, step_selection=[0, 1])
                middle = reader.read(field, step_selection=[1, 1])
                final = reader.read(field, step_selection=[3, 1])
                assert np.allclose(initial, 0.0)
                assert np.max(middle) > 0.0
                assert not np.allclose(middle, final)
                # ParaView requires matching mesh and point-data arrays at
                # every frame, including point IDs and ghost markers.
                for name in ("geometry", field, "vtkOriginalPointIds", "vtkGhostType"):
                    blocks = reader.all_blocks_info(name)
                    assert len(blocks) == 4
                    assert all(len(frame) == 1 for frame in blocks)
                for step in range(4):
                    selection = [step, 1]
                    geometry = reader.read("geometry", step_selection=selection)
                    n_points = geometry.reshape(-1, 3).shape[0]
                    for name in (field, "vtkOriginalPointIds", "vtkGhostType"):
                        values = reader.read(name, step_selection=selection)
                        assert values.size == n_points
                    ids = reader.read("vtkOriginalPointIds", step_selection=selection)
                    assert np.unique(ids).size == n_points
