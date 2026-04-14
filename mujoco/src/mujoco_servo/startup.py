from __future__ import annotations

import os
import shutil
import sys
from typing import Sequence


def maybe_reexec_under_mjpython(argv: Sequence[str]) -> None:
    if sys.platform != "darwin":
        return
    if os.environ.get("MJPYTHON_BIN") or os.environ.get("MJPYTHON_LIBPYTHON"):
        return
    if os.environ.get("MUJOCO_SERVO_DISABLE_MJPYTHON_REEXEC") == "1":
        return
    if not argv:
        return
    command = argv[0]
    if command not in {"sim", "camera"}:
        return
    if "--no-display" in argv:
        return
    mjpython = shutil.which("mjpython")
    if mjpython is None:
        return
    os.execvp(mjpython, ["mjpython", "-m", "mujoco_servo", *list(argv)])
