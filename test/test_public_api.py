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
    "festim_microstructure.ebsd",
    "festim_microstructure.exports",
    "festim_microstructure.formats",
]

#: Modules that must not be imported by ``import festim_microstructure``.
HEAVY_AT_IMPORT = ("dolfinx", "festim", "ufl", "gmsh", "petsc4py", "mpi4py")

SRC = pathlib.Path(festim_microstructure.__file__).parent


@pytest.mark.parametrize("package", PACKAGES)
def test_lazy_exports_have_matching_static_declarations(package):
    """Editors must see the same concrete definitions as runtime callers."""
    module = importlib.import_module(package)
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    declarations = {}
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                for alias in statement.names:
                    assert alias.asname == alias.name, "Use explicit public re-exports"
                    declarations[alias.name] = "." * statement.level + (
                        statement.module or ""
                    )

    expected = dict(module._NAMES)
    expected.update(dict.fromkeys(module._SUBMODULES, "."))
    assert declarations == expected


def test_installed_package_declares_inline_types():
    """Verify installed package declares inline types."""
    assert (SRC / "py.typed").is_file()


@pytest.mark.parametrize("package", PACKAGES)
def test_every_advertised_name_resolves(stubbed_heavy_deps, package):
    """Verify every advertised name resolves."""
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
    """Verify dir advertises exactly all."""
    module = importlib.import_module(package)
    assert dir(module) == sorted(module.__all__)


@pytest.mark.parametrize("package", PACKAGES)
def test_unknown_attributes_raise_attribute_error(package):
    """Verify unknown attributes raise attribute error."""
    module = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no attribute"):
        module.definitely_not_a_real_name


def test_submodules_resolve_as_attributes_of_the_package():
    """``import festim_microstructure`` then ``fm.voronoi``.

    A plain ``__all__`` of submodule-name strings made star-imports work but
    left attribute access failing, because nothing had imported the submodule.
    """
    import festim_microstructure as fm

    assert fm.voronoi.VoronoiMicrostructure is fm.VoronoiMicrostructure
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
    """Verify touching a pure name does not import fenics."""
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
    sorted(
        p for p in SRC.glob("*.py") if p.name not in {"_version.py", "materials.py"}
    ),
    ids=lambda p: p.name,
)
def test_top_level_modules_carry_no_fenics_dependency(path):
    """Only materials.py at the package root needs the solver stack."""
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


def test_standalone_tools_with_solver_imports_blocked():
    """Missing solver packages must not prevent conversion or fm-check startup."""
    probe = """
import importlib.abc
import sys

class NoSolvers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dolfinx', 'festim', 'ufl', 'gmsh',
                                     'mpi4py', 'petsc4py'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoSolvers())
import festim_microstructure as fm
from festim_microstructure.check import _version, main
from festim_microstructure.ebsd.convert import CtfConversion
from festim_microstructure.formats import ctf, msh4, provenance, tesr
from festim_microstructure.meshing.diagnostics import overlay
assert callable(main)
assert callable(CtfConversion)
assert callable(overlay)
assert _version('dolfinx')[0] is None
assert fm.VoronoiMicrostructure is fm.voronoi.VoronoiMicrostructure
try:
    fm.Grain
except ImportError:
    pass
else:
    raise AssertionError('solver exports must still require the solver stack')
"""
    subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True)


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
