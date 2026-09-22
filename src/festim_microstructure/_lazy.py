"""PEP 562 lazy attribute access for the package ``__init__`` modules.

The package has a split personality: ``voronoi.geometry2d`` and the
whole EBSD converter are pure NumPy/SciPy, while ``model``, ``fem`` and
``meshing.ebsd`` need DOLFINx and FESTIM. A top-level namespace built
the usual way -- ``from .model import build`` in ``__init__.py`` --
would drag FEniCS into ``import festim_microstructure``, so a user who only
wants to convert an EBSD map could not import the package at all without it.

:func:`lazy_namespace` gives the same flat namespace without that cost. The
names are *declared* eagerly (they are in ``__all__``, they show up in
``dir()``, and a stale one is caught by the test suite) but the module holding
each is imported on first access.

Each lazy namespace also declares explicit re-exports under ``TYPE_CHECKING``
in its ``__init__.py``. Keep these in sync with the runtime table so editors
can resolve classes, signatures and submodules without executing this helper.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping

__all__ = ["lazy_namespace"]


def lazy_namespace(
    package: str,
    submodules: Iterable[str] = (),
    names: Mapping[str, str] | None = None,
    eager: Iterable[str] = (),
) -> tuple[Callable[[str], object], Callable[[], list[str]], list[str]]:
    """Build the ``(__getattr__, __dir__, __all__)`` triple for a package.

    Args:
        package: the importing package's ``__name__``.
        submodules: subpackages and modules reachable as attributes. Listing
            ``"meshing"`` makes ``festim_microstructure.meshing`` resolve after
            a bare ``import festim_microstructure``, which plain ``__all__``
            entries never did.
        names: attribute name -> the relative module that defines it, e.g.
            ``{"build": ".model"}``.
        eager: names the caller has already bound itself (``__version__``), so
            they belong in ``__all__`` but must not be looked up lazily.

    Returns:
        ``(__getattr__, __dir__, __all__)``, to be assigned at module level.
    """
    names = dict(names or {})
    submodules = tuple(submodules)
    public = sorted({*submodules, *names, *eager})

    def __getattr__(name: str):
        if name in submodules:
            return importlib.import_module(f".{name}", package)
        target = names.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            return getattr(importlib.import_module(target, package), name)
        except ImportError as exc:
            # The usual cause is a missing optional dependency rather than a
            # wrong name, and "No module named 'dolfinx'" out of an attribute
            # lookup on the package is a confusing place to land.
            raise ImportError(
                f"{package}.{name} lives in {package}{target} and that module "
                f"could not be imported: {exc}"
            ) from exc

    def __dir__() -> list[str]:
        return public

    return __getattr__, __dir__, list(public)
