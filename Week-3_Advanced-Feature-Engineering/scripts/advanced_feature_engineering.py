"""Week 3 wrapper: advanced feature engineering.

This project’s Week 3 outputs are produced by `scripts/week3_sentiment_pipeline.py`
(and related scripts). This wrapper exists to match the required structure.
"""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "scripts/week3_sentiment_pipeline.py"])


if __name__ == "__main__":
    raise SystemExit(main())
