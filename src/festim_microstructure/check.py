"""``fm-check`` reports Python dependencies and the EBSD worker."""

import argparse
import importlib
import os
import sys

from .meshing.upxo import probe_worker, resolve_python

__all__ = ["main"]


def _version(module):
    """Return the installed version of an importable module, if available."""
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return getattr(m, "__version__", "?"), None


def main(argv=None):
    """Print dependency and EBSD worker availability for this installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ebsd",
        action="store_true",
        help="require a compatible, working UPXO/DefDAP EBSD stack",
    )
    args = parser.parse_args(argv)
    ok = True
    print("python-side stack")
    for module in ("festim_microstructure", "numpy", "scipy", "matplotlib"):
        v, err = _version(module)
        print(f"  {module:22s} {v if err is None else 'MISSING  ' + err}")
        ok &= err is None
    for module in ("mpi4py", "dolfinx", "festim", "gmsh"):
        v, err = _version(module)
        label = "gmsh (python-gmsh)" if module == "gmsh" else module
        print(f"  {label:22s} {v if err is None else 'MISSING  ' + err}")
        ok &= err is None
    print(f"  prefix                 {sys.prefix}")

    upxo_python = resolve_python(required=False)
    print(f"  EBSD interpreter       {upxo_python or 'not configured'}")
    if upxo_python is not None:
        try:
            report = probe_worker(upxo_python, health=True)
            print("  EBSD worker            OK (runtime imports verified)")
            for name, version in report["packages"].items():
                print(f"    {name:20s} {version}")
        except RuntimeError as exc:
            print(f"  EBSD worker            FAILED: {exc}")
            ok = False
    if upxo_python is None:
        ok &= not (args.ebsd or os.environ.get("FM_UPXO_PYTHON"))
        print("  EBSD meshing: create and activate environment.yml")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
