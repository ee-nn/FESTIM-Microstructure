"""Reusable transport-model building blocks for FESTIM microstructures.

``fisher``      one lattice + one network, kinetic exchange (short circuit)
``resolved``    one subdomain per grain, coupled through the network
``properties``  physics and per-grain / per-boundary coefficient fields

Research workflows, validation studies, publication reproductions and their
figures deliberately live under :mod:`examples`, rather than in the installed
package API.
"""

__all__ = [
    "fisher",
    "properties",
    "resolved",
]
