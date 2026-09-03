"""Shared holdings, Yahoo-price, and return utilities for ETF analyses."""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_YFINANCE_CODE_DIR = Path(r"C:\python\code")
ISHARES_TO_YAHOO = {"BRKB": "BRK-B", "BFB": "BF-B"}


@dataclass
class HoldingsUniverse:
    path: Path
    stem: str
    stocks: list[str]
    display_tickers: dict[str, str]
    sectors: dict[str, str]
    holdings_weights: dict[str, float]


def yahoo_symbol(symbol: str) -> str:
    """Translate common holdings-file share-class notation for Yahoo."""
    normalized = symbol.strip().upper().replace(".", "-")
    return ISHARES_TO_YAHOO.get(normalized, normalized)


def read_symbols_file(path: Path) -> list[str]:
    """Read symbols from a CSV symbol/ticker column or one-per-line text file."""
    if not path.is_file():
        raise FileNotFoundError(f"symbols file not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        column = columns.get("symbol", columns.get("ticker"))
        if column is None:
            raise ValueError(
                f"CSV symbols file must contain a 'symbol' or 'ticker' column: {path}"
            )
        return [
            value.strip()
            for value in frame[column].dropna().astype(str)
            if value.strip()
        ]
    with path.open(encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle if line.strip()]


def infer_price_file_symbols(path: Path) -> list[str]:
    """Read symbol columns, excluding the leading date/index column."""
    if not path.is_file():
        raise FileNotFoundError(f"price file not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if len(columns) < 2:
        raise ValueError(f"price file has no symbol columns: {path}")
    symbols = [
        yahoo_symbol(str(column))
        for column in columns[1:]
        if str(column).strip()
    ]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError(f"price file has no usable symbol columns: {path}")
    return symbols


def symbol_stem(symbols: list[str]) -> str:
    """Build a bounded-length deterministic filename stem from symbols."""
    joined = "__".join(symbol.replace("^", "") for symbol in symbols)
    if len(joined) <= 120:
        return joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"{joined[:100]}__{digest}"


def read_equity_holdings(path: Path) -> pd.DataFrame:
    """Read an iShares-style CSV, locating its header rather than assuming a row."""
    if not path.is_file():
        raise FileNotFoundError(f"holdings file not found: {path}")

    header_row = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle):
            first_cell = line.split(",", 1)[0].strip().strip('"')
            if first_cell.casefold() in {"ticker", "symbol"}:
                header_row = number
                break
    if header_row is None:
        raise ValueError(f"could not find a Ticker or Symbol header in {path}")

    holdings = pd.read_csv(path, skiprows=header_row)
    ticker_col = next(
        (column for column in holdings if column.casefold() in {"ticker", "symbol"}),
        None,
    )
    if ticker_col is None:
        raise ValueError(f"no ticker column in {path}")
    if "Asset Class" in holdings.columns:
        holdings = holdings[
            holdings["Asset Class"].astype(str).str.strip().str.casefold().eq("equity")
        ]

    sector_col = next(
        (column for column in holdings if column.casefold() == "sector"), None
    )
    result = pd.DataFrame(
        {"ticker": holdings[ticker_col].astype(str).str.strip().str.upper()}
    )
    if sector_col is None:
        result["sector"] = "Unclassified"
    else:
        sectors = holdings[sector_col].astype(str).str.strip()
        result["sector"] = sectors.mask(
            sectors.str.casefold().isin(["", "-", "nan"]), "Unclassified"
        )
    weight_col = next(
        (
            column
            for column in holdings
            if column.casefold() in {"weight (%)", "weight", "weight%"}
        ),
        None,
    )
    if weight_col is None:
        result["holdings_weight"] = np.nan
    else:
        result["holdings_weight"] = pd.to_numeric(
            holdings[weight_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce",
        )
    result = result[~result["ticker"].isin(["", "-", "NAN"])]
    result["yahoo_symbol"] = result["ticker"].map(yahoo_symbol)
    return result.drop_duplicates("yahoo_symbol", keep="first").reset_index(drop=True)


def prepare_holdings(
    paths: list[Path],
) -> tuple[list[HoldingsUniverse], list[str], str]:
    """Load funds and return funds, their deduplicated stock union, and shared stem."""
    funds: list[HoldingsUniverse] = []
    stems: list[str] = []
    for path in paths:
        holdings = read_equity_holdings(path)
        if holdings.empty:
            raise ValueError(f"no equity constituents found in {path}")
        if path.stem in stems:
            raise ValueError(
                "holdings files must have unique filename stems for output naming: "
                f"{path.stem}"
            )
        stems.append(path.stem)
        funds.append(
            HoldingsUniverse(
                path=path,
                stem=path.stem,
                stocks=holdings["yahoo_symbol"].tolist(),
                display_tickers=dict(
                    zip(holdings["yahoo_symbol"], holdings["ticker"])
                ),
                sectors=dict(zip(holdings["yahoo_symbol"], holdings["sector"])),
                holdings_weights=dict(
                    zip(holdings["yahoo_symbol"], holdings["holdings_weight"])
                ),
            )
        )
    stocks = list(dict.fromkeys(stock for fund in funds for stock in fund.stocks))
    shared_stem = stems[0] if len(stems) == 1 else "__".join(stems)
    return funds, stocks, shared_stem


def download_prices(
    symbols: list[str], days: int, end_date: str | None, code_dir: Path
) -> pd.DataFrame:
    """Use the supplied Yahoo helper, requesting a generous calendar-day buffer."""
    code_dir = code_dir.resolve()
    if not (code_dir / "yfinance_util.py").is_file():
        raise FileNotFoundError(f"yfinance_util.py not found in {code_dir}")
    sys.path.insert(0, str(code_dir))
    from yfinance_util import get_historical_prices  # type: ignore

    if end_date:
        end = pd.Timestamp(end_date)
    else:
        # Yahoo's end is exclusive, avoiding an incomplete current session.
        end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=max(365, days * 2))
    prices = get_historical_prices(
        symbols,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        field="Adj Close",
    )
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    return (
        prices.sort_index()
        .reindex(columns=symbols)
        .apply(pd.to_numeric, errors="coerce")
    )


def read_saved_prices(path: Path) -> pd.DataFrame:
    """Read every column from a saved adjusted-close CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"saved price file not found: {path}")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.columns = [yahoo_symbol(str(column)) for column in prices.columns]
    duplicate_columns = prices.columns[prices.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            "duplicate symbols in saved price file after Yahoo normalization: "
            + ", ".join(duplicate_columns)
        )
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index().apply(pd.to_numeric, errors="coerce")


def load_saved_prices(path: Path, symbols: list[str]) -> pd.DataFrame:
    """Load a saved adjusted-close CSV and validate its required symbols."""
    prices = read_saved_prices(path)
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    if missing:
        raise ValueError("saved price file is missing symbols: " + ", ".join(missing))
    return prices.reindex(columns=symbols)


def ensure_saved_symbols(
    path: Path,
    symbols: list[str],
    days: int,
    code_dir: Path,
    no_download_missing: bool,
    critical_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], float]:
    """Load a price cache and fetch required symbol columns while preserving data."""
    if path.is_file():
        prices = read_saved_prices(path)
        yahoo_end = (prices.index.max().normalize() + pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ) if not prices.index.empty else None
    else:
        if no_download_missing:
            raise FileNotFoundError(f"saved price file not found: {path}")
        prices = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        yahoo_end = None

    critical = set(critical_symbols or [])
    # Existing unusable noncritical candidates are handled by coverage filtering.
    missing = [symbol for symbol in symbols if symbol not in prices.columns]
    missing.extend(
        symbol
        for symbol in critical
        if symbol in prices.columns and prices[symbol].notna().sum() < 2
        and symbol not in missing
    )
    if not missing:
        return prices, [], [], 0.0
    if no_download_missing:
        raise ValueError(
            "saved price file is missing usable data for symbols: "
            + ", ".join(missing)
        )

    download_start = time.perf_counter()
    downloaded = download_prices(missing, days, yahoo_end, code_dir)
    download_elapsed = time.perf_counter() - download_start
    merged_index = prices.index.union(downloaded.index).sort_values()
    prices = prices.reindex(merged_index)
    for symbol in missing:
        downloaded_series = downloaded[symbol].reindex(merged_index)
        if symbol in prices.columns:
            prices[symbol] = prices[symbol].combine_first(downloaded_series)
        else:
            prices[symbol] = downloaded_series
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date")

    still_missing = [symbol for symbol in missing if prices[symbol].notna().sum() < 2]
    critical_failures = [symbol for symbol in still_missing if symbol in critical]
    if critical_failures:
        raise RuntimeError(
            "Yahoo Finance returned insufficient data for symbols: "
            + ", ".join(critical_failures)
        )
    return prices, missing, still_missing, download_elapsed


def ensure_saved_history(
    path: Path,
    symbols: list[str],
    return_days: int,
    code_dir: Path,
    no_download_missing: bool,
    critical_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], float]:
    """Expand a backtest cache when possible, returning all available history."""
    prices, attempted, unavailable, download_elapsed = ensure_saved_symbols(
        path,
        symbols,
        return_days,
        code_dir,
        no_download_missing,
        critical_symbols,
    )
    required_price_rows = return_days + 1
    if len(prices) >= required_price_rows:
        return prices, attempted, unavailable, download_elapsed
    if no_download_missing:
        return prices, attempted, unavailable, download_elapsed

    download_start = time.perf_counter()
    downloaded = download_prices(symbols, return_days, None, code_dir)
    download_elapsed += time.perf_counter() - download_start
    merged_index = prices.index.union(downloaded.index).sort_values()
    prices = prices.reindex(merged_index)
    for symbol in symbols:
        downloaded_series = downloaded[symbol].reindex(merged_index)
        if symbol in prices.columns:
            prices[symbol] = prices[symbol].combine_first(downloaded_series)
        else:
            prices[symbol] = downloaded_series
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date")

    attempted = list(dict.fromkeys([*attempted, *symbols]))
    unavailable = [symbol for symbol in symbols if prices[symbol].notna().sum() < 2]
    critical = set(critical_symbols or [])
    critical_failures = [symbol for symbol in unavailable if symbol in critical]
    if critical_failures:
        raise RuntimeError(
            "Yahoo Finance returned insufficient data for symbols: "
            + ", ".join(critical_failures)
        )
    return prices, attempted, unavailable, download_elapsed


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns without filling missing price observations."""
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
