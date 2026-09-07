"""Solver settings that the unscaled, many-subdomain problems need.

Both of these are workarounds and both are worth knowing about; see
``docs/gb_homogenisation.md`` ("Numerics worth knowing") for the full story.
"""

from petsc4py import PETSc

__all__ = ["ATOL", "DIRECT_SOLVER_OPTIONS", "tune_direct_solver"]

# on the Newton residual. Way below the FESTIM default because nothing here is scaled
# NOTE nondimensionalization is the better fix
ATOL = 1e-25
"""Absolute tolerance on the Newton residual; see ``docs/gb_homogenisation.md``
for why the default stalls silently on an unscaled problem."""

DIRECT_SOLVER_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 400,  # % headroom over the estimated workspace
}
"""FESTIM's direct solve with a larger MUMPS working array.

One subdomain per grain makes a wide block system whose fill-in MUMPS routinely
under-estimates, and it fails with ``INFOG(1)=-9`` rather than reallocating. How
much fill-in there is depends on the pivoting, so the same mesh can factor at
one exchange rate and run out of memory at another.
"""


def tune_direct_solver(model, icntl_14=400):
    """Give MUMPS more room for fill-in, after the solver exists.

    One subdomain per grain makes a wide block system, and the blocks are scaled
    very differently -- a lattice stiffness of order ``D_bulk`` against an
    exchange term of order ``k/delta``. MUMPS then needs more working memory
    than it estimated and errors with ``INFOG(1) = -9`` instead of reallocating.

    This cannot be done through ``petsc_options`` because FESTIM deletes them from
    the database as soon as the solver is built (a workaround for PETSc issue
    1201). Writing it back afterwards, under the solver's own prefix, ensures it
    is read at the right moment.

    Call it after ``initialise()`` and before the first solve;
    :meth:`~festim_microstructure.models.resolved.MicroModel.run` already does.
    """
    solver = getattr(model, "solver", None)
    snes = getattr(solver, "solver", None)
    if snes is None:
        return
    prefix = snes.getOptionsPrefix() or ""
    # only under the solver's own prefix: an unprefixed copy is never read, and
    # PETSc reports it as an unused option at the end of the run
    PETSc.Options()[f"{prefix}mat_mumps_icntl_14"] = icntl_14
