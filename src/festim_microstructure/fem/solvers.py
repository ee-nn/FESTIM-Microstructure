"""Solver workarounds for many-subdomain problems."""

from petsc4py import PETSc

__all__ = ["tune_direct_solver"]


def tune_direct_solver(model, icntl_14=400):
    """Give MUMPS ``icntl_14`` percent of workspace headroom.

    Call it between ``model.initialise()`` and ``model.run()``. One subdomain
    per grain makes a wide, badly scaled block system whose fill-in MUMPS
    routinely under-estimates, and it stops with ``INFOG(1) = -9`` at some
    parameter values and not others on the same mesh.

    The fix is the PETSc option ``mat_mumps_icntl_14``, but it cannot be passed
    through the problem's ``petsc_options``: FESTIM, like dolfinx's
    ``NonlinearProblem`` underneath it, deletes every option it was given from
    the PETSc database as soon as the solver is built, whereas a ``mat_mumps_*``
    option is only read at ``PCSetUp``, during the first solve. This writes the
    option back under the solver's own prefix, where MUMPS will find it. The
    default solver -- MUMPS behind ``preonly``/``lu`` -- is FESTIM's own, so
    nothing else needs setting.
    """
    solver = getattr(model, "solver", None)
    snes = getattr(solver, "solver", None)
    if snes is None:
        return
    prefix = snes.getOptionsPrefix() or ""
    # MUMPS reads only the solver-prefixed key.
    PETSc.Options().setValue(f"{prefix}mat_mumps_icntl_14", icntl_14)
