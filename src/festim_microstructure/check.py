"""``fm-check``: report what this installation can and cannot do.

Prints the Python-side stack (dolfinx, festim, gmsh API, mpi4py) and where the
external programs resolve to. Exit status is non-zero only if the FEniCSx side
is broken; a missing Neper is reported but is not an error, since the Voronoi
route and the EBSD converter work without it.
"""

import importlib
import os
import re
import subprocess
import sys

from ._binaries import ENV_VARS, resolve_all

__all__ = ["main"]


def _version(module):
    try:
        m = importlib.import_module(module)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return getattr(m, "__version__", "?"), None


_VERSION_RE = re.compile(r"\d+\.\d+")


def _program_version(path, flag):
    """First output line that looks like a version.

    Both streams are scanned, because POV-Ray writes its banner to stderr and
    precedes it with a harmless "cannot open the user configuration file
    ~/.povray/3.7/povray.conf" line when that optional file is absent.
    """
    try:
        out = subprocess.run(
            [path, flag], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not run ({type(e).__name__})"
    lines = [ln.strip() for ln in (out.stdout + "\n" + out.stderr).splitlines()]
    lines = [ln for ln in lines if ln]
    for ln in lines:
        if _VERSION_RE.search(ln) and "cannot open" not in ln:
            return ln
    return lines[0] if lines else f"exit {out.returncode}"


def main(argv=None):
    ok = True
    print("python-side stack")
    for module in ("festim_microstructure", "numpy", "scipy", "matplotlib", "PIL"):
        v, err = _version(module)
        print(f"  {module:22s} {v if err is None else 'MISSING  ' + err}")
        ok &= err is None
    for module in ("mpi4py", "dolfinx", "festim", "gmsh"):
        v, err = _version(module)
        label = "gmsh (python-gmsh)" if module == "gmsh" else module
        print(f"  {label:22s} {v if err is None else 'MISSING  ' + err}")
        ok &= err is None
    print(f"  prefix                 {sys.prefix}")

    print("\nexternal programs (explicit path > FM_*_BIN > PATH)")
    found = resolve_all()
    for name, var in ENV_VARS.items():
        path = found[name]
        env_val = os.environ.get(var)
        source = "env var" if env_val and path == env_val else "PATH"
        if path is None:
            print(
                f"  {name:8s} not found      ({var} unset)"
                if not env_val
                else f"  {name:8s} not found      ({var}={env_val!r} does not exist)"
            )
        elif str(path).startswith("INVALID"):
            print(f"  {name:8s} {path}")
            ok = False
        else:
            flag = {"neper": "--version", "gmsh": "--version", "povray": "--version"}[
                name
            ]
            print(f"  {name:8s} {path}  [{source}]  {_program_version(path, flag)}")
    if found["neper"] is None:
        print(
            "\nNeper is not available: Neper-backed tessellations and EBSD meshing "
            "will not run.\nThe Voronoi (Gmsh) route and the EBSD .ctf converter do "
            "not need it. To add it:\n"
            "  conda env create -f environment-neper.yml && tools/link-neper-env.sh"
        )
    elif found["gmsh"] is None:
        print(
            "\nNeper was found but no gmsh executable: `neper -M` will fail. "
            "Add gmsh to the Neper environment and re-run tools/link-neper-env.sh."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
