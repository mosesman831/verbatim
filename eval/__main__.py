"""Make ``python -m eval`` run the harness CLI."""

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
