import os

import pytest


@pytest.fixture
def ctf_writer():
    """Create complete Oxford text maps with optional quality/phase fields."""

    def write(path, euler, spacing=(1.0, 1.0), mad=None, phases=None, reverse=False):
        import numpy as np

        euler = np.asarray(euler)
        ny, nx, _ = euler.shape
        phases = np.ones((ny, nx), dtype=int) if phases is None else np.asarray(phases)
        mad = np.full((ny, nx), 0.1) if mad is None else np.asarray(mad)
        header = (
            f"Channel Text File\nXCells\t{nx}\nYCells\t{ny}\n"
            f"XStep\t{spacing[0]}\nYStep\t{spacing[1]}\n"
            "Phases\t2\n1;1;1\t90;90;90\tCubic\t11\t229\n1;1;1\t90;90;90\tOtherCubic\t11\t229\n"
            "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n"
        )
        rows = []
        for y in range(ny):
            for x in range(nx):
                values = [
                    phases[y, x],
                    x * spacing[0],
                    y * spacing[1],
                    8,
                    0,
                    *euler[y, x],
                    mad[y, x],
                    100,
                    100,
                ]
                rows.append("\t".join(str(v) for v in values) + "\n")
        path.write_text(header + "".join(reversed(rows) if reverse else rows))
        return path

    return write


def pytest_sessionfinish(session, exitstatus):
    """Required full-stack CI must not silently pass with missing dependencies."""
    if os.environ.get("FM_REQUIRE_FULL_STACK") == "1":
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter and reporter.stats.get("skipped"):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
