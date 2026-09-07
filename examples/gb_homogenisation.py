"""Identify an anisotropic effective diffusivity for the grain-boundary network.

Thin wrapper over :func:`festim_microstructure.models.homogenisation.main`;
see ``docs/gb_homogenisation.md``. Run::

    python examples/gb_homogenisation.py --sizes 2e-6 3e-6 4e-6 --out rve.json
    python examples/gb_homogenisation.py --k-sweep 1e-6 1e-4 1e-2 3 --out rve.json
    mpirun -n 8 python examples/gb_homogenisation.py
"""

from festim_microstructure.models.homogenisation import main

if __name__ == "__main__":
    main()
