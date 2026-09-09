"""Diaz-Rodriguez et al. (2022) OKMC permeation setup, as a FESTIM continuum model.

Thin wrapper over :func:`festim_microstructure.models.diaz_rodriguez.main`; the
parameters and their provenance are documented in that module. Run::

    python examples/diaz_rodriguez.py
"""

from festim_microstructure.models.diaz_rodriguez import main

if __name__ == "__main__":
    main()
