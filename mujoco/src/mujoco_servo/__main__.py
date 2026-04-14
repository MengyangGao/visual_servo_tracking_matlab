import sys

from .startup import maybe_reexec_under_mjpython

maybe_reexec_under_mjpython(sys.argv[1:])

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
