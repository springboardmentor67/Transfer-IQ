"""Week 3 wrapper: sentiment analysis.

Implemented in `scripts/week3_sentiment_pipeline.py`.
This wrapper exists to match the required deliverable structure.
"""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "scripts/week3_sentiment_pipeline.py"])


if __name__ == "__main__":
    raise SystemExit(main())
