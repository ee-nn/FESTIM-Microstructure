"""Does the identified tensor actually reproduce the microstructure?

Thin wrapper over :func:`festim_microstructure.models.validation.main`. Run::

    python examples/gb_validation.py --out validation.json
"""

from festim_microstructure.models.validation import main

if __name__ == "__main__":
    main()
