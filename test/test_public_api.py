"""The declared public API, and the layering that keeps it cheap to import.

Two things are checked here. First, that every name a package advertises in
``__all__`` actually resolves -- lazy attribute access means a typo in the
name-to-module table is invisible until someone touches that one name, so the
table is walked in full. Second, that importing the package does not drag in
FEniCS, which is the property the laziness exists to preserve.
"""

import ast
import importlib
import pathlib
import subprocess
import sys

import pytest

import festim_microstructure

PACKAGES = [
    "festim_microstructure",
    "festim_microstructure.fem",
    "festim_microstructure.meshing",
    "festim_microstructure.meshing.ebsd",
    "festim_microstructure.models",
    "festim_microstructure.postprocessing",
]

#: Modules that must not be imported by ``import festim_microstructure``.
HEAVY_AT_IMPORT = ("dolfinx", "festim", "ufl", "gmsh", "petsc4py", "numpy")

SRC = pathlib.Path(festim_microstructure.__file__).parent


@pytest.mark.parametrize("package", PACKAGES)
def test_every_advertised_name_resolves(stubbed_heavy_deps, package):
    module = importlib.import_module(package)
    unresolved = []
    for name in module.__all__:
        try:
            getattr(module, name)
        except (AttributeError, ImportError) as exc:
            unresolved.append(f"{name}: {exc}")
    assert unresolved == []


@pytest.mark.parametrize("package", PACKAGES)
def test_dir_advertises_exactly_all(stubbed_heavy_deps, package):
    module = importlib.import_module(package)
    assert dir(module) == sorted(module.__all__)


@pytest.mark.parametrize("package", PACKAGES)
def test_unknown_attributes_raise_attribute_error(package):
    module = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no attribute"):
        module.definitely_not_a_real_name


def test_submodules_resolve_as_attributes_of_the_package():
    """``import festim_microstructure`` then ``fm.meshing.voronoi``.

    A plain ``__all__`` of submodule-name strings made star-imports work but
    left attribute access failing, because nothing had imported the submodule.
    """
    import festim_microstructure as fm

    assert fm.meshing.voronoi.VoronoiMicrostructure is fm.VoronoiMicrostructure
    assert fm.microstructure.MeshedMicrostructure is fm.MeshedMicrostructure


def test_importing_the_package_pulls_in_nothing_heavy():
    """Run in a subprocess: other tests have already imported half of this."""
    probe = (
        "import sys; import festim_microstructure; "
        f"print(','.join(m for m in {HEAVY_AT_IMPORT!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_touching_a_pure_name_does_not_import_fenics():
    probe = (
        "import sys, festim_microstructure as fm; fm.VoronoiMicrostructure; "
        "print('dolfinx' in sys.modules or 'festim' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def _module_level_imports(path):
    """Root package names imported at module scope.

    Only ``tree.body`` is walked: an import inside a function is deferred on
    purpose and is exactly what the layering allows.
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
    sorted(p for p in SRC.glob("*.py") if p.name != "_version.py"),
    ids=lambda p: p.name,
)
def test_top_level_modules_carry_no_fenics_dependency(path):
    """The layering ``fem/`` exists to enforce.

    Stated as a negative so that adding a pure module needs no edit here:
    anything at the top level of the package must be importable without
    DOLFINx or FESTIM. Glue that cannot be belongs in ``fem/``.
    """
    forbidden = {"dolfinx", "festim", "ufl", "basix", "petsc4py", "mpi4py"}
    assert _module_level_imports(path) & forbidden == set()


def test_fem_is_where_the_fenics_glue_lives():
    fem = SRC / "fem"
    assert fem.is_dir()
    heavy = {
        p.name
        for p in fem.glob("*.py")
        if _module_level_imports(p) & {"dolfinx", "festim", "petsc4py"}
    }
    assert heavy == {"solvers.py", "subdomains.py"}
