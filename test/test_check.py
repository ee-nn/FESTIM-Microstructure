"""EBSD diagnostics must report failures and provide a useful exit status."""

import pytest

from festim_microstructure import check


@pytest.mark.parametrize(
    "available,failed,required,status",
    [
        (False, False, False, 0),
        (False, False, True, 1),
        (True, False, True, 0),
        (True, True, True, 1),
    ],
)
def test_ebsd_health_status(monkeypatch, capsys, available, failed, required, status):
    monkeypatch.delenv("FM_UPXO_PYTHON", raising=False)
    monkeypatch.setattr(check, "_version", lambda name: ("test", None))
    monkeypatch.setattr(
        check, "resolve_python", lambda **kw: "/worker/python" if available else None
    )
    monkeypatch.setattr(
        check, "resolve_all", lambda: {name: None for name in check.ENV_VARS}
    )

    def probe(executable, **kwargs):
        assert executable == "/worker/python"
        assert kwargs["health"]
        if failed:
            raise RuntimeError("UPXO requires evaluated UPXO 1.3.1")
        return {"packages": {"upxo": "1.3.1", "defdap": "0.93.6"}}

    monkeypatch.setattr(check, "probe_worker", probe)
    assert check.main(["--ebsd"] if required else []) == status
    output = capsys.readouterr().out
    if available:
        assert ("FAILED" if failed else "runtime imports verified") in output
    else:
        assert "not configured" in output


def test_explicit_missing_worker_fails_default_check(monkeypatch):
    monkeypatch.setenv("FM_UPXO_PYTHON", "/does/not/exist")
    monkeypatch.setattr(check, "_version", lambda name: ("test", None))
    monkeypatch.setattr(check, "resolve_python", lambda **kw: None)
    monkeypatch.setattr(
        check, "resolve_all", lambda: {name: None for name in check.ENV_VARS}
    )
    assert check.main([]) == 1
