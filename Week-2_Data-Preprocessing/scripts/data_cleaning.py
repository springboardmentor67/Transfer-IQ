"""Week 2 wrapper: data cleaning.

This repo’s Week 2 preprocessing is implemented in `scripts/week2_data_processing.py`.
This wrapper exists to match the required deliverable structure.
"""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "scripts/week2_data_processing.py"])


if __name__ == "__main__":
    raise SystemExit(main())
