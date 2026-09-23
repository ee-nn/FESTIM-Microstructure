"""The prepared EBSD mesh keeps its tags and SI scale across cached exports."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_fenics


@requires_fenics
def test_pipeline_exports_si_mesh(tmp_path, monkeypatch):
    import gmsh

    from festim_microstructure.formats.msh4 import read_mesh
    from festim_microstructure.meshing import ebsd

    executable = shutil.which("gmsh")
    if executable is None:
        pytest.skip("needs the Gmsh executable")
    raw = tmp_path / "poly-unscaled"
    gmsh.initialize()
    try:
        gmsh.model.add("grain")
        surface = gmsh.model.occ.addRectangle(0, 0, 0, 2, 3)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        for _, tag in gmsh.model.getEntities(1):
            gmsh.model.addPhysicalGroup(1, [tag], tag)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(raw) + ".msh4")
    finally:
        gmsh.finalize()
    np.savetxt(str(raw) + ".sttesr", [[2, 2, 3, 0.5, 0.5]])
    np.savetxt(str(raw) + "-grainori.txt", [[0.1, 0.2, 0.3]])

    def mesh_tesr(tesr, options):
        assert options.base == raw
        return raw

    monkeypatch.setattr(ebsd, "mesh_tesr", mesh_tesr)
    monkeypatch.setattr(ebsd, "mesh_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ebsd.TesrMeshOptions, "binaries", lambda self: (None, executable, None)
    )
    options = ebsd.EbsdOptions(tesr="unused.tesr", unit=1e-6, check_images=False)
    for _ in range(2):
        base = ebsd.run_ebsd_pipeline(options, workdir=tmp_path, force=False)
        assert base == tmp_path / "poly"
        assert not (tmp_path / "raw").exists()
        mesh, cells, facets = read_mesh(base, gdim=2)
        np.testing.assert_allclose(mesh.geometry.x.max(axis=0), [2e-6, 3e-6, 0])
        np.testing.assert_array_equal(np.unique(cells.values), [1])
        np.testing.assert_array_equal(np.unique(facets.values), [1, 2, 3, 4])
        np.testing.assert_allclose(ebsd.read_extent(base), [2e-6, 3e-6])
        np.testing.assert_allclose(
            np.loadtxt(str(base) + "-grainori.txt"), [0.1, 0.2, 0.3]
        )

    extra = tmp_path / "poly-unscaled-extra.txt"
    extra.write_text("temporary Neper output")
    removed = ebsd.cleanup_unscaled_files(base)
    assert set(removed) == {
        raw.with_suffix(".msh4"),
        Path(str(raw) + ".sttesr"),
        Path(str(raw) + "-grainori.txt"),
        extra,
    }
    assert not list(tmp_path.glob("poly-unscaled*"))
    assert base.with_suffix(".msh4").is_file()
    assert Path(str(base) + ".sttesr").is_file()
    assert Path(str(base) + "-grainori.txt").is_file()
