"""The figures in ``docs/gb_homogenisation.md``.

Thin wrapper over :func:`festim_microstructure.postprocessing.figures.main`. Run::

    python examples/gb_figures.py --rve rve.json --validation validation.json
"""

from festim_microstructure.postprocessing.figures import main

if __name__ == "__main__":
    main()
