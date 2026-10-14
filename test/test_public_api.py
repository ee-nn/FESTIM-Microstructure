"""The public exports and namespace access provided by ordinary imports."""

import ast
import importlib
import pathlib

import pytest

import festim_microstructure

PACKAGES = [
    "festim_microstructure",
    "festim_microstructure.fem",
    "festim_microstructure.meshing",
    "festim_microstructure.ebsd",
    "festim_microstructure.exports",
    "festim_microstructure.formats",
]

SRC = pathlib.Path(festim_microstructure.__file__).parent


@pytest.mark.parametrize("package", PACKAGES)
def test_every_advertised_name_is_imported(package):
    """Public exports are available immediately in each package namespace."""
    module = importlib.import_module(package)
    assert set(module.__all__) <= vars(module).keys()


@pytest.mark.parametrize("package", PACKAGES)
def test_unknown_attributes_raise_attribute_error(package):
    """Verify unknown attributes raise attribute error."""
    module = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no attribute"):
        module.definitely_not_a_real_name


def test_submodules_resolve_as_attributes_of_the_package():
    """Curated exports and module-qualified names refer to the same objects."""
    import festim_microstructure as fm

    assert fm.voronoi.VoronoiMicrostructure is fm.VoronoiMicrostructure
    assert fm.microstructure.MeshedMicrostructure is fm.MeshedMicrostructure
    assert fm.Grain is fm.fem.Grain is fm.fem.subdomains.Grain
    assert fm.EbsdMicrostructure is fm.meshing.ebsd.EbsdMicrostructure
    assert fm.ebsd.CtfConversion is fm.ebsd.convert.CtfConversion
    assert fm.UpxoMeshOptions is fm.meshing.UpxoMeshOptions


def _module_level_imports(path):
    """Root package names imported at module scope.

    Only ``tree.body`` is walked; function-local imports are excluded.
    """
    roots = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "path",
    sorted(
        p for p in SRC.glob("*.py") if p.name not in {"_version.py", "materials.py"}
    ),
    ids=lambda p: p.name,
)
def test_top_level_solver_imports_are_confined_to_materials(path):
    """Direct solver imports at the package root stay in materials.py."""
    forbidden = {"dolfinx", "festim", "ufl", "basix", "petsc4py", "mpi4py"}
    assert _module_level_imports(path) & forbidden == set()


def test_fem_is_where_the_fenics_glue_lives():
    """Verify fem is where the fenics glue lives."""
    fem = SRC / "fem"
    assert fem.is_dir()
    heavy = {
        p.name
        for p in fem.glob("*.py")
        if _module_level_imports(p) & {"dolfinx", "festim", "petsc4py"}
    }
    assert heavy == {"solvers.py", "subdomains.py"}


def test_formats_do_not_depend_on_science_modules():
    """File I/O must not import segmentation, meshing, or material fields."""
    for path in (SRC / "formats").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    (
                        "festim_microstructure.ebsd",
                        "festim_microstructure.meshing",
                        "festim_microstructure.materials",
                    )
                ), path
