"""Command-line entry point for the mean-variance walk-forward backtest."""

from __future__ import annotations

import sys

from portfolio_backtest import main


if __name__ == "__main__":
    try:
        raise SystemExit(main("mean-variance"))
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
