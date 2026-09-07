"""Locating the external programs (Neper, Gmsh, POV-Ray) without importing dolfinx.

Neper cannot share a conda environment with DOLFINx (conda-forge's ``neper``
pins ``zlib < 1.3`` through ``scotch 6.1``, DOLFINx needs ``libzlib >= 1.3.2``),
so it lives in ``environment-neper.yml`` and is reached by path. Each program
is resolved from, in order:

1. an explicit path passed by the caller;
2. its environment variable -- ``FM_NEPER_BIN``, ``FM_GMSH_BIN``,
   ``FM_POVRAY_BIN`` -- which ``tools/link-neper-env.sh`` records on the
   FEniCSx environment so they are exported on ``conda activate``;
3. ``PATH``.

Kept dependency-free so that the pure-NumPy EBSD converter can use it too.
"""

import os
import shutil
from pathlib import Path

__all__ = ["ENV_VARS", "find_binary", "resolve_all", "subprocess_env"]

ENV_VARS = {"neper": "FM_NEPER_BIN", "gmsh": "FM_GMSH_BIN", "povray": "FM_POVRAY_BIN"}

_INSTALL_HINT = (
    "Install it into its own conda environment with "
    "`conda env create -f environment-neper.yml`, then run "
    "`tools/link-neper-env.sh` from the festim-microstructure environment "
    "(or export {var} yourself, or pass the path explicitly)."
)


def find_binary(name, explicit=None, env_var=None, required=True):
    """Resolve a program: explicit path, then ``env_var``, then ``PATH``.

    ``env_var`` defaults to the entry for ``name`` in :data:`ENV_VARS`.

    Neper re-tokenizes its arguments and splits its input-file field on
    whitespace, so a path with a space in it arrives as several unusable
    fragments; such paths are rejected here rather than later and less clearly.
    """
    if env_var is None:
        env_var = ENV_VARS.get(name)
    candidates = [explicit]
    if env_var:
        candidates.append(os.environ.get(env_var))
    candidates.append(name)
    for c in candidates:
        if not c:
            continue
        if Path(c).is_file():
            path = str(Path(c).resolve())
        else:
            path = shutil.which(c)
        if path is None:
            continue
        if any(ch.isspace() for ch in path):
            raise ValueError(
                f"the path {path!r} for {name} contains whitespace. Neper "
                "re-tokenizes its arguments, so a path with a space in it is "
                "torn into fragments. Move or symlink the binary somewhere "
                "without one."
            )
        return path
    if required:
        raise FileNotFoundError(
            f"{name} not found. " + _INSTALL_HINT.format(var=env_var or "its path")
        )
    return None


def subprocess_env(*binaries, base=None):
    """A copy of the environment with the directories of ``binaries`` first on PATH.

    Neper spawns Gmsh (and, for ``-V``, POV-Ray) by name unless given a path,
    and the two live in an environment that is never activated. Prepending their
    directories is the equivalent of activating it for the child process only.
    """
    env = dict(os.environ if base is None else base)
    dirs = []
    for b in binaries:
        if b:
            d = str(Path(b).parent)
            if d not in dirs:
                dirs.append(d)
    env["PATH"] = os.pathsep.join([*dirs, env.get("PATH", "")])
    return env


def resolve_all():
    """``{name: path-or-None}`` for every program in :data:`ENV_VARS`.

    Never raises: this is for reporting (``fm-check``), not for running.
    """
    out = {}
    for name, var in ENV_VARS.items():
        try:
            out[name] = find_binary(name, env_var=var, required=False)
        except ValueError as e:  # whitespace in the path
            out[name] = f"INVALID: {e}"
    return out
