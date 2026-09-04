"""Estimate ETF risk from current constituent weights and stock covariances."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from etf_data import (
    DEFAULT_YFINANCE_CODE_DIR,
    HoldingsUniverse,
    daily_returns,
    ensure_saved_history,
    prepare_holdings,
    yahoo_symbol,
)
from portfolio_optimization import (
    covariance_matrix,
    covariance_to_correlation,
    holdings_implied_covariance,
    portfolio_risk_contributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current-holdings-implied ETF risk with risk estimated "
            "from historical ETF returns."
        )
    )
    parser.add_argument(
        "holdings", type=Path, nargs="+", help="One or more ETF holdings CSVs"
    )
    parser.add_argument(
        "--etfs", nargs="+", required=True,
        help="ETF Yahoo symbols in the same order as the holdings files",
    )
    parser.add_argument(
        "--days", type=int, nargs="+", default=[252],
        help="One or more daily-return windows (default: 252)",
    )
    parser.add_argument(
        "--covariance", nargs="+", choices=["ledoit-wolf", "sample"],
        default=["sample"],
        help="One or both constituent covariance estimators (default: sample)",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.90,
        help="Minimum return coverage per constituent and ETF (default: 0.90)",
    )
    parser.add_argument(
        "--weight-treatment", choices=["normalized", "as-reported"],
        default="normalized",
        help="Normalize included equity weights or leave them as reported (default: normalized)",
    )
    parser.add_argument(
        "--sector", action="store_true",
        help="Save constituent risk contributions aggregated by sector",
    )
    parser.add_argument(
        "--annualization-days", type=float, default=252.0,
        help="Trading days used to annualize covariance and volatility (default: 252)",
    )
    parser.add_argument(
        "--prices-file", type=Path,
        help="Combined saved-price CSV (default: merged holdings name in --output-dir)",
    )
    parser.add_argument(
        "--no-download-missing", action="store_true",
        help="Use only the saved price file and fail if required data are absent",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory for saved prices and results (default: output)",
    )
    parser.add_argument(
        "--code-dir", type=Path, default=DEFAULT_YFINANCE_CODE_DIR,
        help="Directory containing the Yahoo Finance download helper",
    )
    args = parser.parse_args()
    if len(args.etfs) != len(args.holdings):
        parser.error("--etfs must contain one symbol for each holdings file")
    args.etfs = [yahoo_symbol(symbol) for symbol in args.etfs]
    if len(set(args.etfs)) != len(args.etfs):
        parser.error("--etfs symbols must be unique")
    if any(window < 2 for window in args.days):
        parser.error("every --days value must be at least 2")
    args.days = list(dict.fromkeys(args.days))
    args.covariance = list(dict.fromkeys(args.covariance))
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be greater than 0 and at most 1")
    if not math.isfinite(args.annualization_days) or args.annualization_days <= 0:
        parser.error("--annualization-days must be finite and positive")
    return args


def positive_source_weights(fund: HoldingsUniverse) -> pd.Series:
    weights = pd.Series(fund.holdings_weights, dtype=float) / 100.0
    return weights.where(np.isfinite(weights) & (weights > 0))


def calculation_weights(
    fund: HoldingsUniverse,
    eligible: list[str],
    treatment: str,
) -> tuple[pd.Series, float, float, float]:
    source = positive_source_weights(fund)
    source_total = float(source.sum(skipna=True))
    included_symbols = [symbol for symbol in fund.stocks if symbol in eligible]
    included = source.reindex(included_symbols).dropna()
    included_total = float(included.sum())
    if source_total <= 0:
        raise ValueError(f"{fund.path} has no positive equity holdings weights")
    if included_total <= 0:
        raise ValueError(f"{fund.path} has no positively weighted eligible constituents")
    if treatment == "normalized":
        included = included / included_total
    retained_fraction = included_total / source_total
    return included, source_total, included_total, retained_fraction


def labeled_matrix(values: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=labels, columns=labels)


def matrix_paths(
    output_dir: Path,
    stem: str,
    window: int,
    method: str,
) -> dict[str, Path]:
    prefix = output_dir / f"{stem}_{window}d_{method.replace('-', '_')}"
    return {
        "implied_covariance": Path(f"{prefix}_implied_annualized_covariance.csv"),
        "historical_covariance": Path(f"{prefix}_historical_annualized_covariance.csv"),
        "covariance_difference": Path(f"{prefix}_covariance_difference.csv"),
        "implied_correlation": Path(f"{prefix}_implied_correlation.csv"),
        "historical_correlation": Path(f"{prefix}_historical_correlation.csv"),
        "correlation_difference": Path(f"{prefix}_correlation_difference.csv"),
    }


def risk_contribution_rows(
    fund: HoldingsUniverse,
    etf_symbol: str,
    window: int,
    method: str,
    eligible: list[str],
    constituent_covariance: np.ndarray,
    fund_weights: pd.Series,
    annualization_days: float,
) -> list[dict[str, object]]:
    weight_vector = fund_weights.reindex(eligible, fill_value=0.0).to_numpy()
    marginal, component, fraction = portfolio_risk_contributions(
        constituent_covariance, weight_vector, annualization_days
    )
    eligible_index = {symbol: index for index, symbol in enumerate(eligible)}
    source = positive_source_weights(fund)
    rows: list[dict[str, object]] = []
    for symbol in fund.stocks:
        index = eligible_index.get(symbol)
        included = index is not None and symbol in fund_weights.index
        rows.append(
            {
                "etf": etf_symbol,
                "window_days": window,
                "covariance_method": method,
                "ticker": fund.display_tickers.get(symbol, symbol),
                "yahoo_symbol": symbol,
                "sector": fund.sectors.get(symbol, "Unclassified"),
                "included": included,
                "source_weight": source.get(symbol, np.nan),
                "calculation_weight": fund_weights.get(symbol, 0.0),
                "marginal_annualized_volatility": (
                    marginal[index] if included else np.nan
                ),
                "component_annualized_volatility": (
                    component[index] if included else 0.0
                ),
                "variance_contribution_fraction": (
                    fraction[index] if included else 0.0
                ),
            }
        )
    return rows


def calculate_window(
    returns: pd.DataFrame,
    funds: list[HoldingsUniverse],
    etfs: list[str],
    stocks: list[str],
    window: int,
    method: str,
    coverage: float,
    treatment: str,
    annualization_days: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, pd.DataFrame],
]:
    window_returns = returns.tail(window)
    if len(window_returns) < window:
        raise ValueError(
            f"only {len(window_returns)} return rows are available for the "
            f"{window}-day window"
        )
    required = max(2, math.ceil(window * coverage))
    observations = window_returns.notna().sum()
    positive_symbols = {
        symbol
        for fund in funds
        for symbol, weight in positive_source_weights(fund).items()
        if pd.notna(weight)
    }
    eligible = [
        symbol for symbol in stocks
        if symbol in positive_symbols and observations.get(symbol, 0) >= required
    ]
    if not eligible:
        raise ValueError(
            f"no constituents meet {coverage:.1%} coverage for the {window}-day window"
        )
    deficient_etfs = [
        symbol for symbol in etfs if observations.get(symbol, 0) < required
    ]
    if deficient_etfs:
        raise ValueError(
            f"ETF prices do not meet {coverage:.1%} coverage for the "
            f"{window}-day window: " + ", ".join(deficient_etfs)
        )
    joint_columns = list(dict.fromkeys([*eligible, *etfs]))
    sample = window_returns[joint_columns].dropna()
    if len(sample) < required:
        raise ValueError(
            f"only {len(sample)} common return observations remain for the "
            f"{window}-day window; {required} are required"
        )

    constituent_covariance = covariance_matrix(sample[eligible], method)
    historical_covariance = covariance_matrix(sample[etfs], "sample")
    weight_columns: list[np.ndarray] = []
    fund_weight_series: list[pd.Series] = []
    weight_metadata: list[tuple[float, float, float]] = []
    for fund in funds:
        weights, source_total, included_total, retained = calculation_weights(
            fund, eligible, treatment
        )
        fund_weight_series.append(weights)
        weight_columns.append(weights.reindex(eligible, fill_value=0.0).to_numpy())
        weight_metadata.append((source_total, included_total, retained))
    weight_matrix = np.column_stack(weight_columns)
    implied_covariance = holdings_implied_covariance(
        constituent_covariance, weight_matrix
    )
    implied_correlation = covariance_to_correlation(implied_covariance)
    historical_correlation = covariance_to_correlation(historical_covariance)
    annualized_implied = implied_covariance * annualization_days
    annualized_historical = historical_covariance * annualization_days

    summaries: list[dict[str, object]] = []
    contributions: list[dict[str, object]] = []
    for index, (fund, etf_symbol) in enumerate(zip(funds, etfs)):
        source_total, included_total, retained = weight_metadata[index]
        implied_volatility = math.sqrt(max(0.0, annualized_implied[index, index]))
        historical_volatility = math.sqrt(
            max(0.0, annualized_historical[index, index])
        )
        summaries.append(
            {
                "etf": etf_symbol,
                "holdings_file": str(fund.path),
                "window_days": window,
                "covariance_method": method,
                "historical_covariance_method": "sample",
                "weight_treatment": treatment,
                "constituents": len(fund.stocks),
                "eligible_constituents": int(
                    sum(symbol in fund_weight_series[index].index for symbol in fund.stocks)
                ),
                "union_eligible_constituents": len(eligible),
                "common_observations": len(sample),
                "source_equity_weight": source_total,
                "included_source_weight": included_total,
                "source_weight_retained": retained,
                "calculation_weight_sum": float(fund_weight_series[index].sum()),
                "implied_annualized_volatility": implied_volatility,
                "historical_annualized_volatility": historical_volatility,
                "volatility_difference": implied_volatility - historical_volatility,
                "volatility_ratio": implied_volatility / historical_volatility,
            }
        )
        contributions.extend(
            risk_contribution_rows(
                fund,
                etf_symbol,
                window,
                method,
                eligible,
                constituent_covariance,
                fund_weight_series[index],
                annualization_days,
            )
        )

    matrices = {
        "implied_covariance": labeled_matrix(annualized_implied, etfs),
        "historical_covariance": labeled_matrix(annualized_historical, etfs),
        "covariance_difference": labeled_matrix(
            annualized_implied - annualized_historical, etfs
        ),
        "implied_correlation": labeled_matrix(implied_correlation, etfs),
        "historical_correlation": labeled_matrix(historical_correlation, etfs),
        "correlation_difference": labeled_matrix(
            implied_correlation - historical_correlation, etfs
        ),
    }
    return summaries, contributions, matrices


def print_matrix(title: str, matrix: pd.DataFrame, scale: float = 1.0) -> None:
    print(f"{title}:")
    print((matrix * scale).to_string(float_format=lambda value: f"{value:.3f}"))


def main() -> int:
    overall_start = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    funds, stocks, shared_stem = prepare_holdings(args.holdings)
    symbols = list(dict.fromkeys([*stocks, *args.etfs]))
    prices_path = (
        args.prices_file
        if args.prices_file is not None
        else args.output_dir / f"{shared_stem}_adjusted_close.csv"
    )

    data_start = time.perf_counter()
    prices, downloaded, unavailable, download_elapsed = ensure_saved_history(
        prices_path,
        symbols,
        max(args.days),
        args.code_dir,
        args.no_download_missing,
        critical_symbols=args.etfs,
    )
    prices = prices.reindex(columns=symbols)
    returns = daily_returns(prices).dropna(how="all")
    data_elapsed = time.perf_counter() - data_start

    constituent_prices_path = (
        args.output_dir / f"{shared_stem}_constituent_adjusted_close.csv"
    )
    etf_prices_path = args.output_dir / f"{shared_stem}_etf_adjusted_close.csv"
    prices.reindex(columns=stocks).to_csv(
        constituent_prices_path, index_label="date"
    )
    prices.reindex(columns=args.etfs).to_csv(etf_prices_path, index_label="date")

    calculation_start = time.perf_counter()
    all_summaries: list[dict[str, object]] = []
    all_contributions: list[dict[str, object]] = []
    written_matrix_paths: list[Path] = []
    display_results: list[tuple[int, str, pd.DataFrame, dict[str, pd.DataFrame]]] = []
    for window in args.days:
        for method in args.covariance:
            summaries, contributions, matrices = calculate_window(
                returns,
                funds,
                args.etfs,
                stocks,
                window,
                method,
                args.coverage,
                args.weight_treatment,
                args.annualization_days,
            )
            all_summaries.extend(summaries)
            all_contributions.extend(contributions)
            paths = matrix_paths(args.output_dir, shared_stem, window, method)
            for matrix_name, matrix in matrices.items():
                matrix.to_csv(paths[matrix_name], index_label="etf")
                written_matrix_paths.append(paths[matrix_name])
            display_results.append(
                (window, method, pd.DataFrame(summaries), matrices)
            )

    summary = pd.DataFrame(all_summaries)
    contributions = pd.DataFrame(all_contributions)
    summary_path = args.output_dir / f"{shared_stem}_holdings_risk_summary.csv"
    summary.to_csv(summary_path, index=False)
    contribution_paths: list[Path] = []
    sector_paths: list[Path] = []
    for fund, etf_symbol in zip(funds, args.etfs):
        fund_contributions = contributions.loc[
            contributions["etf"] == etf_symbol
        ].sort_values(
            ["window_days", "covariance_method", "variance_contribution_fraction"],
            ascending=[True, True, False],
        )
        path = args.output_dir / f"{fund.stem}_holding_risk_contributions.csv"
        fund_contributions.to_csv(path, index=False)
        contribution_paths.append(path)
        if args.sector:
            sector = (
                fund_contributions.groupby(
                    ["etf", "window_days", "covariance_method", "sector"],
                    as_index=False,
                )[
                    [
                        "source_weight",
                        "calculation_weight",
                        "component_annualized_volatility",
                        "variance_contribution_fraction",
                    ]
                ]
                .sum()
                .sort_values(
                    ["window_days", "covariance_method", "variance_contribution_fraction"],
                    ascending=[True, True, False],
                )
            )
            sector_path = args.output_dir / f"{fund.stem}_sector_risk_contributions.csv"
            sector.to_csv(sector_path, index=False)
            sector_paths.append(sector_path)
    calculation_elapsed = time.perf_counter() - calculation_start

    print(f"Using prices: {prices_path}")
    if downloaded:
        print("Downloaded symbols: " + ", ".join(downloaded))
    if unavailable:
        print("Unavailable constituents excluded: " + ", ".join(unavailable))
    for window, method, window_summary, matrices in display_results:
        common = int(window_summary["common_observations"].iloc[0])
        union_count = int(window_summary["union_eligible_constituents"].iloc[0])
        print(
            f"\n=== {window}-day {method} constituent covariance: "
            f"{union_count} union constituents, "
            f"{common} common observations ==="
        )
        display = window_summary[
            [
                "etf",
                "implied_annualized_volatility",
                "historical_annualized_volatility",
                "volatility_difference",
                "volatility_ratio",
                "eligible_constituents",
                "constituents",
                "source_weight_retained",
            ]
        ].rename(
            columns={
                "implied_annualized_volatility": "implied_vol",
                "historical_annualized_volatility": "historical_vol",
                "volatility_difference": "difference",
                "volatility_ratio": "ratio",
                "eligible_constituents": "included",
                "source_weight_retained": "weight_retained",
            }
        )
        for column in ["implied_vol", "historical_vol", "difference", "weight_retained"]:
            display[column] = display[column].map(lambda value: f"{100 * value:.2f}%")
        display["ratio"] = display["ratio"].map(lambda value: f"{value:.3f}")
        print(display.to_string(index=False))
        for etf_symbol in args.etfs:
            top_risk = contributions.loc[
                (contributions["etf"] == etf_symbol)
                & (contributions["window_days"] == window)
                & (contributions["covariance_method"] == method)
                & contributions["included"]
            ].nlargest(5, "variance_contribution_fraction")
            risk_display = top_risk[
                [
                    "ticker",
                    "sector",
                    "calculation_weight",
                    "variance_contribution_fraction",
                ]
            ].copy()
            for column in ["calculation_weight", "variance_contribution_fraction"]:
                risk_display[column] = risk_display[column].map(
                    lambda value: f"{100 * value:.2f}%"
                )
            print(f"top {etf_symbol} constituent risk contributions:")
            print(risk_display.to_string(index=False))
        print_matrix(
            "implied annualized covariance (% squared)",
            matrices["implied_covariance"],
            10_000.0,
        )
        print_matrix(
            "historical annualized covariance (% squared)",
            matrices["historical_covariance"],
            10_000.0,
        )
        print_matrix(
            "implied minus historical covariance (% squared)",
            matrices["covariance_difference"],
            10_000.0,
        )
        print_matrix("implied correlation", matrices["implied_correlation"])
        print_matrix("historical correlation", matrices["historical_correlation"])
        print_matrix(
            "implied minus historical correlation",
            matrices["correlation_difference"],
        )

    if downloaded:
        print(f"Wrote {prices_path}")
    for path in [
        constituent_prices_path,
        etf_prices_path,
        summary_path,
        *contribution_paths,
        *sector_paths,
        *written_matrix_paths,
    ]:
        print(f"Wrote {path}")
    print(f"Data elapsed: {data_elapsed:.3f} seconds")
    if downloaded:
        print(f"Download elapsed: {download_elapsed:.3f} seconds")
    print(f"Calculations elapsed: {calculation_elapsed:.3f} seconds")
    print(f"Overall elapsed: {time.perf_counter() - overall_start:.3f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
