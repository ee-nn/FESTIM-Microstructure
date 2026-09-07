"""Locating the external programs (Neper, Gmsh, POV-Ray) without importing dolfinx.

Kept dependency-free so that the pure-NumPy EBSD converter can use it too.
"""

import os
import shutil
from pathlib import Path

__all__ = ["ENV_VARS", "find_binary"]

ENV_VARS = {"neper": "FM_NEPER_BIN", "gmsh": "FM_GMSH_BIN", "povray": "FM_POVRAY_BIN"}


def find_binary(name, explicit=None, env_var=None, required=True):
    """Resolve a program: explicit path, then ``env_var``, then ``PATH``.

    Neper re-tokenizes its arguments and splits its input-file field on
    whitespace, so a path with a space in it arrives as several unusable
    fragments; such paths are rejected here rather than later and less clearly.
    """
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
            f"{name} not found. Install it with `conda install conda-forge::{name}`"
            f" (ideally into its own environment) and either put it on PATH, set "
            f"{env_var}, or pass its path explicitly."
        )
    return None
