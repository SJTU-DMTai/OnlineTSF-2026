"""Run an experiment or add hindsight diagnostics to an existing experiment."""

import sys


if __name__ == "__main__":
    if sys.argv[1:2] == ["diagnose"]:
        from .diagnostics import main

        main(sys.argv[2:])
    else:
        from .experiment import main

        main()
