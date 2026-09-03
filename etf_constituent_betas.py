"""Download ETF constituent prices and estimate daily market betas.

The ``--days`` values are daily *return* windows. Data is retrieved once for the
longest window, and each regression uses up to that window's number of aligned
return observations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from etf_data import (
    DEFAULT_YFINANCE_CODE_DIR,
    daily_returns,
    download_prices,
    load_saved_prices,
    prepare_holdings,
    yahoo_symbol,
)

GRID = [-1.0, 0.0, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute constituent betas to one or more ETFs using Yahoo Finance."
    )
    parser.add_argument(
        "holdings", type=Path, nargs="+",
        help="One or more holdings CSVs (for example TOPT_holdings.csv)",
    )
    parser.add_argument("--etfs", nargs="+", default=["VOO", "XLE"], help="Factor ETF tickers")
    parser.add_argument(
        "--days", type=int, nargs="+", default=[252],
        help="One or more daily-return windows, e.g. --days 63 126 252 (default: 252)",
    )
    parser.add_argument(
        "--end-date",
        help="Optional Yahoo-exclusive end date, YYYY-MM-DD (default: today, excluding a partial session)",
    )
    parser.add_argument(
        "--multiple", action="store_true",
        help="Also regress every stock on all ETFs jointly, including an intercept",
    )
    parser.add_argument(
        "--vol", action="store_true",
        help="Compute annualized historical volatility for every constituent",
    )
    parser.add_argument(
        "--idiovol", nargs="?", const="both",
        choices=["univariate", "multiple", "both"],
        help=(
            "Compute annualized idiosyncratic volatility; defaults to both when "
            "the flag is given without a mode"
        ),
    )
    parser.add_argument(
        "--annualization-days", type=float, default=252.0,
        help="Trading days used to annualize volatility (default: 252)",
    )
    parser.add_argument(
        "--sector", action="store_true",
        help="Add sector columns and print/save statistics grouped by sector",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory for downloaded data and results (default: output)",
    )
    parser.add_argument(
        "--no-save-data", action="store_true",
        help="Do not save the raw downloaded adjusted-close data",
    )
    parser.add_argument(
        "--use-saved-data",
        nargs="?",
        const="",
        metavar="CSV",
        help=(
            "Skip Yahoo and read saved prices instead; with no CSV, use "
            "<output-dir>/<shared-prefix>_adjusted_close.csv"
        ),
    )
    parser.add_argument(
        "--code-dir", type=Path, default=DEFAULT_YFINANCE_CODE_DIR,
        help=r"Directory containing yfinance_util.py (default: C:\python\code)",
    )
    args = parser.parse_args()
    if any(days < 2 for days in args.days):
        parser.error("every --days value must be at least 2")
    if args.annualization_days <= 0:
        parser.error("--annualization-days must be positive")
    args.days = list(dict.fromkeys(args.days))
    if not args.etfs:
        parser.error("at least one ETF is required")
    return args


def univariate_betas(
    returns: pd.DataFrame,
    stocks: list[str],
    etfs: list[str],
    display_tickers: dict[str, str],
    windows: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        for etf in etfs:
            for stock in stocks:
                # Align first, then select the window so every beta uses the latest
                # N dates on which both the stock and factor have valid returns.
                paired = returns[[stock, etf]].dropna().tail(window)
                variance = paired[etf].var(ddof=1)
                beta = paired[stock].cov(paired[etf]) / variance if len(paired) >= 2 and variance > 0 else np.nan
                rows.append(
                    {"window_days": window, "etf": etf,
                     "ticker": display_tickers[stock], "yahoo_symbol": stock,
                     "beta": beta, "observations": len(paired)}
                )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["window_days", "etf", "beta", "ticker"], na_position="last"
    ).reset_index(drop=True)


def multiple_betas(
    returns: pd.DataFrame,
    stocks: list[str],
    etfs: list[str],
    display_tickers: dict[str, str],
    windows: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        for stock in stocks:
            # Multiple regression requires dates shared by the stock and all ETFs.
            sample = returns[[stock, *etfs]].dropna().tail(window)
            row: dict[str, object] = {
                "window_days": window, "ticker": display_tickers[stock],
                "yahoo_symbol": stock, "observations": len(sample),
                "intercept": np.nan, "r_squared": np.nan,
            }
            row.update({f"beta_{etf}": np.nan for etf in etfs})
            if len(sample) > len(etfs) + 1:
                x = np.column_stack([np.ones(len(sample)), sample[etfs].to_numpy()])
                y = sample[stock].to_numpy()
                coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
                if rank == x.shape[1]:
                    fitted = x @ coefficients
                    total_ss = np.sum((y - y.mean()) ** 2)
                    row["intercept"] = coefficients[0]
                    row.update({f"beta_{etf}": value for etf, value in zip(etfs, coefficients[1:])})
                    row["r_squared"] = 1 - np.sum((y - fitted) ** 2) / total_ss if total_ss > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["window_days", "ticker"]).reset_index(drop=True)


def historical_volatilities(
    returns: pd.DataFrame,
    stocks: list[str],
    display_tickers: dict[str, str],
    windows: list[int],
    annualization_days: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    annualizer = np.sqrt(annualization_days)
    for window in windows:
        for stock in stocks:
            sample = returns[stock].dropna().tail(window)
            volatility = sample.std(ddof=1) * annualizer if len(sample) >= 2 else np.nan
            rows.append(
                {"window_days": window, "ticker": display_tickers[stock],
                 "yahoo_symbol": stock, "annualized_volatility": volatility,
                 "observations": len(sample)}
            )
    return pd.DataFrame(rows).sort_values(
        ["window_days", "annualized_volatility", "ticker"], na_position="last"
    ).reset_index(drop=True)


def univariate_idiosyncratic_volatilities(
    returns: pd.DataFrame,
    stocks: list[str],
    etfs: list[str],
    display_tickers: dict[str, str],
    windows: list[int],
    annualization_days: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    annualizer = np.sqrt(annualization_days)
    for window in windows:
        for etf in etfs:
            for stock in stocks:
                sample = returns[[stock, etf]].dropna().tail(window)
                row: dict[str, object] = {
                    "window_days": window, "etf": etf,
                    "ticker": display_tickers[stock], "yahoo_symbol": stock,
                    "annualized_idiosyncratic_volatility": np.nan,
                    "observations": len(sample), "r_squared": np.nan,
                }
                if len(sample) > 2:
                    x = np.column_stack([np.ones(len(sample)), sample[etf].to_numpy()])
                    y = sample[stock].to_numpy()
                    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
                    if rank == 2:
                        residuals = y - x @ coefficients
                        residual_variance = np.sum(residuals**2) / (len(sample) - 2)
                        total_ss = np.sum((y - y.mean()) ** 2)
                        row["annualized_idiosyncratic_volatility"] = (
                            np.sqrt(residual_variance) * annualizer
                        )
                        row["r_squared"] = (
                            1 - np.sum(residuals**2) / total_ss if total_ss > 0 else np.nan
                        )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["window_days", "etf", "annualized_idiosyncratic_volatility", "ticker"],
        na_position="last",
    ).reset_index(drop=True)


def multiple_idiosyncratic_volatilities(
    returns: pd.DataFrame,
    stocks: list[str],
    etfs: list[str],
    display_tickers: dict[str, str],
    windows: list[int],
    annualization_days: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    annualizer = np.sqrt(annualization_days)
    parameter_count = len(etfs) + 1
    for window in windows:
        for stock in stocks:
            sample = returns[[stock, *etfs]].dropna().tail(window)
            row: dict[str, object] = {
                "window_days": window, "ticker": display_tickers[stock],
                "yahoo_symbol": stock,
                "annualized_idiosyncratic_volatility": np.nan,
                "observations": len(sample), "r_squared": np.nan,
            }
            if len(sample) > parameter_count:
                x = np.column_stack([np.ones(len(sample)), sample[etfs].to_numpy()])
                y = sample[stock].to_numpy()
                coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
                if rank == parameter_count:
                    residuals = y - x @ coefficients
                    residual_variance = np.sum(residuals**2) / (len(sample) - parameter_count)
                    total_ss = np.sum((y - y.mean()) ** 2)
                    row["annualized_idiosyncratic_volatility"] = (
                        np.sqrt(residual_variance) * annualizer
                    )
                    row["r_squared"] = (
                        1 - np.sum(residuals**2) / total_ss if total_ss > 0 else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["window_days", "annualized_idiosyncratic_volatility", "ticker"],
        na_position="last",
    ).reset_index(drop=True)


def grid_counts(betas: pd.DataFrame) -> pd.DataFrame:
    edges = [-np.inf, *GRID, np.inf]
    labels = ["< -1", "[-1, 0)", "[0, 1)", "[1, 2)", ">= 2"]
    frames = []
    for (window, etf), group in betas.groupby(["window_days", "etf"], sort=False):
        bins = pd.cut(group["beta"], bins=edges, labels=labels, right=False)
        counts = bins.value_counts(sort=False).reindex(labels, fill_value=0)
        frame = counts.rename_axis("beta_range").reset_index(name="count")
        frame.insert(0, "etf", etf)
        frame.insert(0, "window_days", window)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def print_distribution_summary(
    values_to_summarize: pd.Series, scale: float = 1.0, decimals: int = 3
) -> None:
    """Print labels and values on exactly two aligned lines."""
    labels = ["median", "mean", "sd", "min", "max"]
    finite = pd.to_numeric(values_to_summarize, errors="coerce").dropna() * scale
    values = [
        finite.median(), finite.mean(), finite.std(ddof=1), finite.min(), finite.max()
    ]
    print(" ".join(f"{label:>12}" for label in labels))
    print(" ".join(f"{value:12.{decimals}f}" for value in values))


def add_sectors(frame: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Return a copy with a sector column mapped from Yahoo symbol."""
    result = frame.copy()
    sector_values = result["yahoo_symbol"].map(sectors).fillna("Unclassified")
    insert_at = result.columns.get_loc("yahoo_symbol") + 1
    result.insert(insert_at, "sector", sector_values)
    return result


def summarize_by_sector(
    frame: pd.DataFrame,
    metric: str,
    value_column: str,
    factor_label: str | None = None,
) -> pd.DataFrame:
    """Create long-format cross-sectional sector statistics in source units."""
    group_columns = ["window_days", "sector"]
    if "etf" in frame.columns:
        group_columns.insert(1, "etf")
    summary = (
        frame.groupby(group_columns, dropna=False)[value_column]
        .agg(count="count", median="median", mean="mean", sd="std", min="min", max="max")
        .reset_index()
    )
    if "etf" not in summary.columns:
        summary.insert(1, "etf", factor_label or "")
    summary.insert(0, "metric", metric)
    return summary[
        ["metric", "window_days", "etf", "sector", "count",
         "median", "mean", "sd", "min", "max"]
    ]


def print_sector_summaries(
    frame: pd.DataFrame,
    value_column: str,
    scale: float = 1.0,
    decimals: int = 3,
) -> None:
    for sector, group in frame.groupby("sector", sort=True):
        finite_count = pd.to_numeric(group[value_column], errors="coerce").notna().sum()
        print(f"{sector} (n={finite_count}):")
        print_distribution_summary(
            group[value_column], scale=scale, decimals=decimals
        )


def main() -> int:
    overall_start = time.perf_counter()
    args = parse_args()
    etfs = list(dict.fromkeys(yahoo_symbol(symbol) for symbol in args.etfs))
    funds, stocks, shared_stem = prepare_holdings(args.holdings)
    symbols = list(dict.fromkeys([*stocks, *etfs]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_start = time.perf_counter()
    saved_data_mode = args.use_saved_data is not None
    if saved_data_mode:
        saved_path = (
            Path(args.use_saved_data)
            if args.use_saved_data
            else args.output_dir / f"{shared_stem}_adjusted_close.csv"
        )
        prices = load_saved_prices(saved_path, symbols)
    else:
        prices = download_prices(symbols, max(args.days), args.end_date, args.code_dir)
    data_elapsed = time.perf_counter() - data_start
    if prices.dropna(how="all").empty:
        raise RuntimeError("the price data contains no observations")

    if not args.no_save_data:
        raw_path = args.output_dir / f"{shared_stem}_adjusted_close.csv"
        constituent_prices_path = (
            args.output_dir / f"{shared_stem}_constituent_adjusted_close.csv"
        )
        etf_prices_path = args.output_dir / f"{shared_stem}_etf_adjusted_close.csv"
        if not saved_data_mode:
            prices.to_csv(raw_path, index_label="date")
        prices.reindex(columns=stocks).to_csv(constituent_prices_path, index_label="date")
        prices.reindex(columns=etfs).to_csv(etf_prices_path, index_label="date")

    calculation_start = time.perf_counter()
    returns = daily_returns(prices)
    run_univariate_idiovol = args.idiovol in {"univariate", "both"} or (
        args.idiovol == "multiple" and len(etfs) == 1
    )
    fund_results: list[dict[str, object]] = []
    for fund in funds:
        stem = fund.stem
        fund_stocks = fund.stocks
        display_tickers = fund.display_tickers
        sectors = fund.sectors

        betas = univariate_betas(
            returns, fund_stocks, etfs, display_tickers, args.days
        )
        if args.sector:
            betas = add_sectors(betas, sectors)
        counts = grid_counts(betas)
        beta_path = args.output_dir / f"{stem}_univariate_betas.csv"
        count_path = args.output_dir / f"{stem}_beta_grid_counts.csv"
        betas.to_csv(beta_path, index=False)
        counts.to_csv(count_path, index=False)

        multiple_path = None
        if args.multiple:
            multiple = multiple_betas(
                returns, fund_stocks, etfs, display_tickers, args.days
            )
            if args.sector:
                multiple = add_sectors(multiple, sectors)
            multiple_path = args.output_dir / f"{stem}_multiple_betas.csv"
            multiple.to_csv(multiple_path, index=False)

        volatility_path = None
        volatility = None
        if args.vol:
            volatility = historical_volatilities(
                returns, fund_stocks, display_tickers, args.days,
                args.annualization_days,
            )
            if args.sector:
                volatility = add_sectors(volatility, sectors)
            volatility_path = args.output_dir / f"{stem}_historical_volatilities.csv"
            volatility.to_csv(volatility_path, index=False)

        univariate_idiovol_path = None
        univariate_idiovol = None
        if run_univariate_idiovol:
            univariate_idiovol = univariate_idiosyncratic_volatilities(
                returns, fund_stocks, etfs, display_tickers, args.days,
                args.annualization_days,
            )
            if args.sector:
                univariate_idiovol = add_sectors(univariate_idiovol, sectors)
            univariate_idiovol_path = (
                args.output_dir / f"{stem}_univariate_idiosyncratic_volatilities.csv"
            )
            univariate_idiovol.to_csv(univariate_idiovol_path, index=False)

        multiple_idiovol_path = None
        multiple_idiovol = None
        if args.idiovol in {"multiple", "both"} and len(etfs) > 1:
            multiple_idiovol = multiple_idiosyncratic_volatilities(
                returns, fund_stocks, etfs, display_tickers, args.days,
                args.annualization_days,
            )
            if args.sector:
                multiple_idiovol = add_sectors(multiple_idiovol, sectors)
            multiple_idiovol_path = (
                args.output_dir / f"{stem}_multiple_idiosyncratic_volatilities.csv"
            )
            multiple_idiovol.to_csv(multiple_idiovol_path, index=False)

        sector_statistics_path = None
        sector_statistics = None
        if args.sector:
            sector_frames = [summarize_by_sector(betas, "beta", "beta")]
            if volatility is not None:
                sector_frames.append(
                    summarize_by_sector(
                        volatility, "historical_volatility", "annualized_volatility"
                    )
                )
            if univariate_idiovol is not None:
                sector_frames.append(
                    summarize_by_sector(
                        univariate_idiovol,
                        "univariate_idiosyncratic_volatility",
                        "annualized_idiosyncratic_volatility",
                    )
                )
            if multiple_idiovol is not None:
                sector_frames.append(
                    summarize_by_sector(
                        multiple_idiovol,
                        "multiple_idiosyncratic_volatility",
                        "annualized_idiosyncratic_volatility",
                        factor_label=",".join(etfs),
                    )
                )
            sector_statistics = pd.concat(sector_frames, ignore_index=True).sort_values(
                ["metric", "window_days", "etf", "sector"]
            )
            sector_statistics_path = args.output_dir / f"{stem}_sector_statistics.csv"
            sector_statistics.to_csv(sector_statistics_path, index=False)

        fund_results.append(
            {
                "path": fund.path,
                "stem": fund.stem,
                "stocks": fund.stocks,
                "display_tickers": fund.display_tickers,
                "betas": betas,
                "counts": counts,
                "beta_path": beta_path,
                "count_path": count_path,
                "multiple_path": multiple_path,
                "volatility": volatility,
                "volatility_path": volatility_path,
                "univariate_idiovol": univariate_idiovol,
                "univariate_idiovol_path": univariate_idiovol_path,
                "multiple_idiovol": multiple_idiovol,
                "multiple_idiovol_path": multiple_idiovol_path,
                "sector_statistics": sector_statistics,
                "sector_statistics_path": sector_statistics_path,
            }
        )
    calculation_elapsed = time.perf_counter() - calculation_start

    missing = [symbol for symbol in symbols if prices[symbol].notna().sum() < 2]
    print(
        f"Holdings files: {len(funds)}; merged unique equity constituents: {len(stocks)}; "
        f"factor ETFs: {', '.join(etfs)}"
    )
    if saved_data_mode:
        print(f"Using saved prices: {saved_path}")
    if missing:
        print("Symbols with insufficient price data: " + ", ".join(missing))
    for result in fund_results:
        fund_stocks = result["stocks"]
        betas = result["betas"]
        counts = result["counts"]
        volatility = result["volatility"]
        univariate_idiovol = result["univariate_idiovol"]
        multiple_idiovol = result["multiple_idiovol"]
        assert isinstance(fund_stocks, list)
        assert isinstance(betas, pd.DataFrame) and isinstance(counts, pd.DataFrame)

        print(f"\n=== {result['path']} ===")
        print(f"Equity constituents: {len(fund_stocks)}; factor ETFs: {', '.join(etfs)}")
        print("Requested return windows: " + ", ".join(map(str, args.days)))
        for window in args.days:
            window_betas = betas[betas["window_days"] == window]
            min_observations = int(window_betas["observations"].min())
            max_observations = int(window_betas["observations"].max())
            print(
                f"\n{window}-day window "
                f"(actual observations: {min_observations}-{max_observations})"
            )
            for etf in etfs:
                print(f"\n{etf} univariate beta statistics:")
                beta_selection = window_betas["etf"] == etf
                print_distribution_summary(window_betas.loc[beta_selection, "beta"])
                if args.sector:
                    print("sector statistics:")
                    print_sector_summaries(
                        window_betas.loc[beta_selection], "beta"
                    )
                print("grid counts:")
                selection = (
                    (counts["window_days"] == window) & (counts["etf"] == etf)
                )
                print(counts[selection][["beta_range", "count"]].to_string(index=False))

            if isinstance(volatility, pd.DataFrame):
                selection = volatility["window_days"] == window
                print("\nHistorical volatility statistics (annualized %):")
                print_distribution_summary(
                    volatility.loc[selection, "annualized_volatility"],
                    scale=100.0, decimals=2,
                )
                if args.sector:
                    print("sector statistics:")
                    print_sector_summaries(
                        volatility.loc[selection], "annualized_volatility",
                        scale=100.0, decimals=2,
                    )

            if isinstance(univariate_idiovol, pd.DataFrame):
                for etf in etfs:
                    selection = (
                        (univariate_idiovol["window_days"] == window)
                        & (univariate_idiovol["etf"] == etf)
                    )
                    print(
                        f"\n{etf} univariate idiosyncratic volatility "
                        "statistics (annualized %):"
                    )
                    print_distribution_summary(
                        univariate_idiovol.loc[
                            selection, "annualized_idiosyncratic_volatility"
                        ],
                        scale=100.0, decimals=2,
                    )
                    if args.sector:
                        print("sector statistics:")
                        print_sector_summaries(
                            univariate_idiovol.loc[selection],
                            "annualized_idiosyncratic_volatility",
                            scale=100.0, decimals=2,
                        )

            if isinstance(multiple_idiovol, pd.DataFrame):
                selection = multiple_idiovol["window_days"] == window
                factors = ", ".join(etfs)
                print(
                    "\nMultiple idiosyncratic volatility statistics "
                    f"({factors}; annualized %):"
                )
                print_distribution_summary(
                    multiple_idiovol.loc[
                        selection, "annualized_idiosyncratic_volatility"
                    ],
                    scale=100.0, decimals=2,
                )
                if args.sector:
                    print("sector statistics:")
                    print_sector_summaries(
                        multiple_idiovol.loc[selection],
                        "annualized_idiosyncratic_volatility",
                        scale=100.0, decimals=2,
                    )

        print(f"\nWrote {result['beta_path']} and {result['count_path']}")
        for path_key in [
            "multiple_path", "volatility_path", "univariate_idiovol_path",
            "multiple_idiovol_path", "sector_statistics_path",
        ]:
            if result[path_key]:
                print(f"Wrote {result[path_key]}")

    if not args.no_save_data:
        if not saved_data_mode:
            print(f"Wrote {raw_path}")
        print(f"Wrote {constituent_prices_path}")
        print(f"Wrote {etf_prices_path}")
    data_action = "load" if saved_data_mode else "download"
    print(f"Data {data_action} elapsed: {data_elapsed:.3f} seconds")
    print(f"Calculations elapsed: {calculation_elapsed:.3f} seconds")
    print(f"Overall elapsed: {time.perf_counter() - overall_start:.3f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
