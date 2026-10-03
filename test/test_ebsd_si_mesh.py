"""Exercise the real UPXO worker and its tagged DOLFINx reconstruction."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_fenics


@requires_fenics
def test_pipeline_exports_si_mesh_and_preserves_enclosed_grain(
    tmp_path, monkeypatch, ctf_writer
):
    from festim_microstructure.ebsd.orientation import euler_bunge_to_quat
    from festim_microstructure.ebsd.settings import Settings
    from festim_microstructure.formats.ebsd import read_ebsd
    from festim_microstructure.formats.msh4 import read_mesh
    from festim_microstructure.meshing import ebsd, upxo

    if upxo.resolve_python(required=False) is None:
        pytest.skip("configure FM_UPXO_PYTHON for the UPXO integration test")
    labels = np.ones((12, 16), dtype=np.int32)
    labels[4:8, 6:10] = 2
    orientations = np.zeros((*labels.shape, 3))
    orientations[labels == 1] = [0, 30, 0]
    orientations[labels == 2] = [0, 0, 20]
    ctf = ctf_writer(tmp_path / "map.ctf", orientations, spacing=(0.5, 0.75))
    options = ebsd.EbsdOptions(
        ctf=str(ctf),
        unit=1e-6,
        check_images=False,
        import_settings=Settings(ctf=str(ctf), min_pixels=1),
        mesh=upxo.UpxoMeshOptions(smoothing="none"),
    )
    base = ebsd.run_ebsd_pipeline(options, workdir=tmp_path, force=False)
    mesh, cells, facets = read_mesh(base, gdim=2)
    np.testing.assert_allclose(mesh.geometry.x.max(axis=0), [8e-6, 9e-6, 0])
    np.testing.assert_array_equal(np.unique(cells.values), [1, 2])
    np.testing.assert_array_equal(np.unique(facets.values), [1, 2])
    np.testing.assert_allclose(ebsd.read_extent(base), [8e-6, 9e-6])
    np.testing.assert_allclose(
        np.abs(read_ebsd(f"{base}-ebsd.npz").grain_quats),
        np.abs(
            euler_bunge_to_quat(np.array([0, 0]), np.array([30, 0]), np.array([0, 20]))
        ),
        atol=1e-6,
    )
    micro = ebsd.EbsdMicrostructure.from_mesh(
        base, mesh, cells, facets, ebsd.read_extent(base)
    )
    assert micro.n_grains == 2
    assert micro.triple_junctions == 0
    assert micro.surface_edges == 1
    assert micro.interior_mask.sum() == 1
    np.testing.assert_allclose(sorted(micro.edges["length"]), [10e-6, 34e-6])
    micro.check_orientations()
    assert json.loads(Path(f"{base}-validation.json").read_text())["accepted"]
    original = base.with_suffix(".msh4").read_bytes()

    with monkeypatch.context() as context:

        def unexpected_worker(*args, **kwargs):
            raise AssertionError("the matching cache must not rerun UPXO")

        context.setattr(upxo.subprocess, "run", unexpected_worker)
        assert ebsd.run_ebsd_pipeline(options, workdir=tmp_path, force=False) == base
    assert base.with_suffix(".msh4").read_bytes() == original

    options.unit = 1e-3
    ebsd.run_ebsd_pipeline(options, workdir=tmp_path, force=False)
    scaled, _, _ = read_mesh(base, gdim=2)
    np.testing.assert_allclose(scaled.geometry.x.max(axis=0), [8e-3, 9e-3, 0])
    np.testing.assert_allclose(ebsd.read_extent(base), [8e-3, 9e-3])
    assert not list(tmp_path.glob("*.tesr"))
    assert not list(tmp_path.glob("*.sttesr"))
    assert not list(tmp_path.glob("poly-unscaled*"))
    assert base.with_suffix(".msh4").is_file()
    assert Path(f"{base}-validation.json").is_file()
