"""festim-microstructure: microstructure-resolved hydrogen transport with FESTIM.

    import festim as F
    import festim_microstructure as fm

Subpackages
-----------
``fm.meshing``         Voronoi (Gmsh) and Neper polycrystals, EBSD map import.
``fm.models``          Fisher short circuit, the per-grain resolved model,
                       and their coefficient fields.
``fm.postprocessing``  Inventories, network topology checks and measures.

Complete research workflows and publication reproductions are kept in the
repository's ``examples/`` directory, outside the installed API.

The heavy dependencies (dolfinx, festim, gmsh) are imported lazily by the
submodules that need them, so ``import festim_microstructure`` and the pure
NumPy EBSD converter (``fm.meshing.ebsd.ctf``) work without them.
"""

try:
    from ._version import __version__
except ImportError:  # not installed (running from a checkout without pip -e)
    __version__ = "0+unknown"

__all__ = ["__version__"]
