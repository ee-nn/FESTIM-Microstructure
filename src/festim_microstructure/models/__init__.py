"""Transport models on a grain-boundary network.

``fisher``          one lattice + one network, kinetic exchange (short circuit)
``resolved``        one subdomain per grain, coupled through the network
``homogenisation``  identify an anisotropic D_eff from the resolved model
``validation``      check that D_eff predicts what it was not fitted to
``properties``      Physics, per-grain and per-boundary coefficient fields
``diaz_rodriguez``  the Diaz-Rodriguez et al. (2022) OKMC permeation setup
"""

__all__ = [
    "diaz_rodriguez",
    "diaz_rodriguez_nondim",
    "fisher",
    "homogenisation",
    "properties",
    "resolved",
    "validation",
]
