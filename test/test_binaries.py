"""find_binary / subprocess_env / resolve_all, with fake executables."""

import os
import stat

import pytest

from festim_microstructure import _binaries as B


def _fake(dirpath, name):
    p = dirpath / name
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_explicit_path_wins(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _fake(tmp_path / "a", "neper")
    b = _fake(tmp_path / "b", "neper")
    monkeypatch.setenv("FM_NEPER_BIN", str(b))
    assert B.find_binary("neper", explicit=str(a)) == str(a.resolve())


def test_env_var_beats_path(tmp_path, monkeypatch):
    (tmp_path / "env").mkdir()
    (tmp_path / "path").mkdir()
    e = _fake(tmp_path / "env", "neper")
    _fake(tmp_path / "path", "neper")
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    monkeypatch.setenv("FM_NEPER_BIN", str(e))
    assert B.find_binary("neper") == str(e.resolve())


def test_env_var_name_defaults_from_table(tmp_path, monkeypatch):
    g = _fake(tmp_path, "gmsh")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("FM_GMSH_BIN", str(g))
    assert B.find_binary("gmsh") == str(g.resolve())


def test_missing_required_raises_with_install_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("FM_NEPER_BIN", raising=False)
    with pytest.raises(FileNotFoundError, match="environment-neper.yml"):  # noqa: RUF043
        B.find_binary("neper")
    assert B.find_binary("neper", required=False) is None


def test_whitespace_in_path_is_rejected(tmp_path, monkeypatch):
    d = tmp_path / "my envs"
    d.mkdir()
    n = _fake(d, "neper")
    with pytest.raises(ValueError, match="whitespace"):
        B.find_binary("neper", explicit=str(n))


def test_subprocess_env_prepends_unique_dirs(tmp_path):
    env = B.subprocess_env(
        str(tmp_path / "x" / "neper"),
        str(tmp_path / "x" / "gmsh"),
        None,
        str(tmp_path / "y" / "povray"),
        base={"PATH": "/usr/bin", "HOME": "/h"},
    )
    parts = env["PATH"].split(os.pathsep)
    assert parts == [str(tmp_path / "x"), str(tmp_path / "y"), "/usr/bin"]
    assert env["HOME"] == "/h"


def test_resolve_all_never_raises(tmp_path, monkeypatch):
    d = tmp_path / "bad dir"
    d.mkdir()
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("FM_NEPER_BIN", str(_fake(d, "neper")))
    monkeypatch.delenv("FM_GMSH_BIN", raising=False)
    monkeypatch.delenv("FM_POVRAY_BIN", raising=False)
    found = B.resolve_all()
    assert found["neper"].startswith("INVALID")
    assert found["gmsh"] is None and found["povray"] is None
