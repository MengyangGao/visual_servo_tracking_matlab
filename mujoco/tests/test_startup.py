from __future__ import annotations

from mujoco_servo import startup


def test_maybe_reexec_under_mjpython_triggers_for_display_commands(monkeypatch) -> None:
    called: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(startup.sys, "platform", "darwin")
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/mjpython" if name == "mjpython" else None)
    monkeypatch.setattr(startup.os, "execvp", lambda file, args: called.append((file, list(args))))

    startup.maybe_reexec_under_mjpython(["sim", "--task", "t2-fixed"])

    assert called == [("/usr/bin/mjpython", ["mjpython", "-m", "mujoco_servo", "sim", "--task", "t2-fixed"])]


def test_maybe_reexec_under_mjpython_skips_no_display(monkeypatch) -> None:
    called: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(startup.sys, "platform", "darwin")
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/mjpython" if name == "mjpython" else None)
    monkeypatch.setattr(startup.os, "execvp", lambda file, args: called.append((file, list(args))))

    startup.maybe_reexec_under_mjpython(["camera", "--no-display"])

    assert called == []


def test_maybe_reexec_under_mjpython_skips_when_already_in_mjpython(monkeypatch) -> None:
    called: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(startup.sys, "platform", "darwin")
    monkeypatch.setenv("MJPYTHON_BIN", "/usr/bin/mjpython")
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/bin/mjpython" if name == "mjpython" else None)
    monkeypatch.setattr(startup.os, "execvp", lambda file, args: called.append((file, list(args))))

    startup.maybe_reexec_under_mjpython(["sim", "--task", "t2-fixed"])

    assert called == []
