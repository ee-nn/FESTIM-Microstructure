"""Exercise the consolidated study, including real solves and saved-data plots."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_fenics

pytestmark = requires_fenics


@pytest.fixture
def study(monkeypatch, tmp_path):
    """Load the example study with an isolated output directory."""
    path = Path(__file__).parents[1] / "examples" / "gb_homogenisation.py"
    spec = importlib.util.spec_from_file_location("gb_study", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    return module


@pytest.mark.fenics
def test_complete_study_reuses_identification(study, monkeypatch, tmp_path):
    """Verify complete study reuses identification."""
    solves = []
    original = study.CellProblem.solve

    def tracked_solve(cp):
        """Record each cell-problem solve before delegating to the real solver."""
        solves.append(cp.transport)
        return original(cp)

    monkeypatch.setattr(study.CellProblem, "solve", tracked_solve)
    study.main(
        [
            "--sizes",
            "1e-6",
            "1.2e-6",
            "--grain-size",
            "0.6e-6",
            "--aspect",
            "1",
            "--cells-per-grain",
            "3",
            "--steps",
            "3",
            "--window-fraction",
            "1",
            "--temperature",
            "600",
            "--crystal-anisotropy",
            "1.4",
            "--k-sweep",
            "3",
        ]
    )
    record = json.loads((tmp_path / "identified.json").read_text())
    validation = json.loads((tmp_path / "validation.json").read_text())
    # Four fitting solves and two permeation solves; no refitting for validation,
    # no solves for field maps, and k=3 reuses the baseline identification.
    assert len(solves) == 6
    assert all(t.T == 600 and t.crystal_anisotropy == 1.4 for t in solves)
    assert validation["identification"] == record["identifications"][0]
    assert record["k_sweep"][0] == record["identifications"][0]
    assert len(validation["permeation"]) == 8
    for key in ("microstructure", "homogeneous"):
        inventory = np.asarray(validation["uptake"][key])
        assert len(inventory) == 4
        assert inventory[0] == 0
        assert inventory[-1] > 0
        assert np.all(np.diff(inventory) >= 0)
    for name in ("anisotropy", "rve", "sweep", "validation", "microstructure"):
        assert (tmp_path / f"fig_{name}.png").stat().st_size > 1000

    def unexpected_mesh(*args, **kwargs):
        """Fail if the test unexpectedly rebuilds a mesh."""
        pytest.fail("plot-only must not create a mesh or run a simulation")

    monkeypatch.setattr(study, "make_microstructure", unexpected_mesh)
    study.main(["--plot-only"])
    assert len(solves) == 6


@pytest.mark.parametrize(
    "arguments",
    [
        ["--out", "../outside.json"],
        ["--validation-out", "identified.json"],
        ["--window-fraction", "0"],
        ["--steps", "0"],
        ["--plot-only", "--skip-figures"],
    ],
)
def test_invalid_options_fail_before_solving(study, arguments):
    """Verify invalid options fail before solving."""
    with pytest.raises(SystemExit, match="2"):
        study.main(arguments)
