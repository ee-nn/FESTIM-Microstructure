"""Solver settings for unscaled, many-subdomain problems."""

from petsc4py import PETSc

__all__ = ["ATOL", "DIRECT_SOLVER_OPTIONS", "tune_direct_solver"]

# Unscaled problems need a tighter residual tolerance.
ATOL = 1e-25
"""Absolute tolerance on the Newton residual; see ``docs/gb_homogenisation.md``
for why the default stalls silently on an unscaled problem."""

DIRECT_SOLVER_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 400,  # % headroom over the estimated workspace
}
"""FESTIM direct-solver options with extra MUMPS workspace."""


def tune_direct_solver(model, icntl_14=400):
    """Set MUMPS workspace after FESTIM creates the prefixed solver.

    The setting is read during ``PCSetUp``, after FESTIM has consumed
    ``petsc_options``.
    """
    solver = getattr(model, "solver", None)
    snes = getattr(solver, "solver", None)
    if snes is None:
        return
    prefix = snes.getOptionsPrefix() or ""
    # MUMPS reads only the solver-prefixed key.
    PETSc.Options()[f"{prefix}mat_mumps_icntl_14"] = icntl_14
