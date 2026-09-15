"""StatFile / NeperMesh on a hand-written stat set (no Neper, no DOLFINx)."""

import numpy as np
import pytest

from festim_microstructure.formats.msh4 import StatFile
from festim_microstructure.meshing.neper import NeperMesh, NeperSettings


def _write(tmp_path):
    base = tmp_path / "poly"
    # domface theta area zmin zmax
    np.savetxt(
        str(base) + ".stface",
        [[-1, 5.0, 0.2, 0.5, 1.0], [-1, 30.0, 0.3, 0.0, 0.6], [2, 40.0, 0.1, 0.0, 1.0]],
    )
    # domtype facenb length
    np.savetxt(str(base) + ".stedge", [[-1, 3, 0.4], [1, 2, 0.5]])
    # domtype edgenb
    np.savetxt(str(base) + ".stver", [[-1, 4], [0, 3]])
    return base


def test_statfile_rejects_drifted_keys(tmp_path):
    base = _write(tmp_path)
    with pytest.raises(ValueError):
        StatFile(str(base) + ".stface", ("a", "b"))


def test_network_mask_and_junction_depth(tmp_path):
    micro = NeperMesh.from_base(_write(tmp_path), NeperSettings(theta_min=10.0))
    assert list(micro.network_ids) == [2]  # face 1 is low-angle, face 3 on the wall
    assert np.isclose(micro.network_measure, 0.3)
    assert micro.triple_lines == (1, 0.4)
    assert micro.quadruple_points == 1
    # face 2 is the only network face; it does not reach z = 1, so no boundary
    # touches the charged face and the junction-only depth is the surface itself
    assert micro.junction_only_below(1.0) == 1.0
    micro.settings.theta_min = 0.0
    assert micro.junction_only_below(1.0) == 0.5


def test_a_mesh_needs_either_a_cell_count_or_a_base(tmp_path):
    """The two ways in are generating one and reading one back; neither is default."""
    with pytest.raises(TypeError, match="cell count"):
        NeperMesh()


def test_the_stats_are_readable_without_the_solver_stack(tmp_path):
    """Nothing before .mesh touches DOLFINx, so a tessellation can be inspected
    on a machine that cannot run the model."""
    micro = NeperMesh.from_base(_write(tmp_path))
    assert micro.report().startswith("microstructure: 3 faces")
    assert micro.n_cells is None  # read back, so the cell count is not known
    assert "_read" not in vars(micro)  # the mesh has not been touched
