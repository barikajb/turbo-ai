"""Run TURBO from a source checkout: python turbo.py ... (installed users just run `turbo`)."""
import sys

from turbo_core.cli import main

if __name__ == "__main__":
    sys.exit(main())
