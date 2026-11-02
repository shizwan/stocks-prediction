# Contributing

Thank you for your interest in OpenRange Backtest. The project is a research prototype, not financial advice.

## Before proposing changes

Open an issue describing the bug, feature, or reproducibility problem. For behavioral changes, include a minimal reproducible sample using synthetic OHLCV candles and explain the expected result.

## Local checks

1. Create and activate a Python 3.11+ virtual environment.
2. Run `python -m pip install -r requirements.txt`.
3. Run `python -m unittest discover -s tests -v`.
4. Run `python -m compileall -q src`.

Keep market-data requests out of unit tests; use deterministic fixtures. For changes to trade calculations or calendar logic, add regression tests for missing candles, timezone boundaries, and invalid prices.

## Pull requests

Describe the problem, approach, test coverage, and any changed modeling assumptions. Do not commit API keys, market-data licenses, or customer/personal data. Contributions are reviewed before merging.

**Licensing:** The maintainer must select and publish a project-level OSI-approved license after confirming ownership and third-party rights. Until then, do not assume the source is licensed for redistribution or reuse.
