"""StatFile / NeperMicrostructure on a hand-written stat set (no Neper needed)."""

import numpy as np
import pytest

from festim_microstructure.meshing.neper import NeperMicrostructure, StatFile


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
    micro = NeperMicrostructure(_write(tmp_path), theta_min=10.0)
    assert list(micro.network_face_ids) == [
        2
    ]  # face 1 is low-angle, face 3 on the wall
    assert np.isclose(micro.network_area, 0.3)
    assert micro.triple_lines == (1, 0.4)
    assert micro.quadruple_points == 1
    # face 2 is the only network face; it does not reach z = 1, so no boundary
    # touches the charged face and the junction-only depth is the surface itself
    assert micro.junction_only_below(1.0) == 1.0
    micro.theta_min = 0.0
    assert micro.junction_only_below(1.0) == 0.5
