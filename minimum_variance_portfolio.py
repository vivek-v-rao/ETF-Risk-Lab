"""Construct long-only minimum-variance portfolios from saved ETF price data."""

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
    daily_returns,
    ensure_saved_symbols,
    prepare_holdings,
    yahoo_symbol,
)
from portfolio_optimization import completion_covariances, optimize_long_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct long-only minimum-variance constituent portfolios."
    )
    parser.add_argument("holdings", type=Path, nargs="+", help="One or more holdings CSVs")
    parser.add_argument(
        "--prices-file", type=Path,
        help="Saved adjusted-close CSV (default: merged file in --output-dir)",
    )
    parser.add_argument(
        "--days", type=int, nargs="+", default=[252],
        help="One or more daily-return windows (default: 252)",
    )
    parser.add_argument(
        "--covariance", choices=["ledoit-wolf", "sample"], default="ledoit-wolf",
        help="Covariance estimator (default: ledoit-wolf)",
    )
    parser.add_argument(
        "--max-weight", type=float, default=1.0,
        help="Maximum weight per constituent, between 0 and 1 (default: 1)",
    )
    parser.add_argument(
        "--portfolio-level", choices=["stocks", "sectors", "both"], default="stocks",
        help="Optimize stocks, fixed-composition sector sleeves, or both (default: stocks)",
    )
    parser.add_argument(
        "--max-sector-weight", type=float, default=1.0,
        help="Maximum optimized sector-sleeve weight, between 0 and 1 (default: 1)",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.90,
        help="Minimum fraction of window returns required per stock (default: 0.90)",
    )
    parser.add_argument(
        "--annualization-days", type=float, default=252.0,
        help="Trading days used to annualize volatility (default: 252)",
    )
    parser.add_argument(
        "--legacy-symbol",
        help="Fixed legacy ETF position; optimize only the remaining new-money sleeve",
    )
    parser.add_argument(
        "--legacy-weight", type=float,
        help="Fixed legacy fraction when --legacy-symbol is used (default: 0.90)",
    )
    parser.add_argument(
        "--legacy-prices-file", type=Path,
        help="Optional separate saved-price CSV containing the legacy symbol",
    )
    parser.add_argument(
        "--legacy-positions", type=Path,
        help="CSV with ticker and signed fixed portfolio weight columns",
    )
    parser.add_argument(
        "--no-download-missing", action="store_true",
        help="Fail instead of downloading required symbols absent from saved data",
    )
    parser.add_argument(
        "--code-dir", type=Path, default=DEFAULT_YFINANCE_CODE_DIR,
        help=r"Directory containing yfinance_util.py (default: C:\python\code)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory for inputs by default and portfolio outputs (default: output)",
    )
    args = parser.parse_args()
    if any(window < 2 for window in args.days):
        parser.error("every --days value must be at least 2")
    args.days = list(dict.fromkeys(args.days))
    if not 0 < args.max_weight <= 1:
        parser.error("--max-weight must be greater than 0 and at most 1")
    if not 0 < args.max_sector_weight <= 1:
        parser.error("--max-sector-weight must be greater than 0 and at most 1")
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be greater than 0 and at most 1")
    if args.annualization_days <= 0:
        parser.error("--annualization-days must be positive")
    if args.legacy_positions is not None and args.legacy_symbol is not None:
        parser.error("--legacy-positions cannot be combined with --legacy-symbol")
    if args.legacy_positions is not None and args.legacy_weight is not None:
        parser.error("--legacy-positions cannot be combined with --legacy-weight")
    if args.legacy_symbol is None and args.legacy_weight is not None:
        parser.error("--legacy-weight requires --legacy-symbol")
    if (
        args.legacy_prices_file is not None
        and args.legacy_symbol is None
        and args.legacy_positions is None
    ):
        parser.error(
            "--legacy-prices-file requires --legacy-symbol or --legacy-positions"
        )
    if args.legacy_symbol is not None:
        args.legacy_symbol = yahoo_symbol(args.legacy_symbol)
        if args.legacy_weight is None:
            args.legacy_weight = 0.90
        if not 0 <= args.legacy_weight < 1:
            parser.error("--legacy-weight must be at least 0 and less than 1")
    else:
        args.legacy_weight = 0.0
    return args


def read_legacy_positions(path: Path) -> dict[str, float]:
    """Read signed fixed weights; their net sum determines the new-money fraction."""
    if not path.is_file():
        raise FileNotFoundError(f"legacy positions file not found: {path}")
    positions = pd.read_csv(path)
    ticker_column = next(
        (
            column
            for column in positions
            if column.casefold() in {"ticker", "symbol"}
        ),
        None,
    )
    weight_column = next(
        (column for column in positions if column.casefold() == "weight"), None
    )
    if ticker_column is None or weight_column is None:
        raise ValueError("legacy positions CSV requires ticker and weight columns")
    tickers = positions[ticker_column].astype(str).map(yahoo_symbol)
    weights = pd.to_numeric(positions[weight_column], errors="coerce")
    if tickers.isin(["", "NAN", "-"]).any() or weights.isna().any():
        raise ValueError("legacy positions CSV contains an invalid ticker or weight")
    if tickers.duplicated().any():
        duplicates = tickers[tickers.duplicated(keep=False)].unique().tolist()
        raise ValueError("duplicate legacy position tickers: " + ", ".join(duplicates))
    result = {
        ticker: float(weight)
        for ticker, weight in zip(tickers, weights)
        if float(weight) != 0.0
    }
    if not result:
        raise ValueError("legacy positions CSV contains no nonzero positions")
    net_weight = sum(result.values())
    if net_weight >= 1:
        raise ValueError(
            f"legacy position net weight must be less than 1; got {net_weight:.6f}"
        )
    return result


def optimize_fund_window(
    returns: pd.DataFrame,
    stocks: list[str],
    display_tickers: dict[str, str],
    sectors: dict[str, str],
    holdings_weights: dict[str, float],
    window: int,
    covariance_method: str,
    max_weight: float,
    coverage: float,
    annualization_days: float,
    legacy_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    legacy_weights = legacy_weights or {}
    overlaps = sorted(set(legacy_weights) & set(stocks))
    if overlaps:
        raise ValueError(
            "legacy positions cannot also be candidate constituents: "
            + ", ".join(overlaps)
        )
    data_symbols = [*stocks, *legacy_weights]
    window_returns = returns.reindex(columns=data_symbols).dropna(how="all").tail(window)
    required_observations = max(2, math.ceil(window * coverage))
    observation_counts = window_returns.notna().sum()
    eligible = [
        symbol
        for symbol in stocks
        if observation_counts.get(symbol, 0) >= required_observations
    ]
    if not eligible:
        raise ValueError(
            f"no constituents meet {coverage:.1%} coverage for the {window}-day window"
        )

    # Complete dates produce a valid joint covariance matrix. Coverage filtering
    # removes unavailable/new securities before the common-date intersection.
    sample_columns = [*eligible, *legacy_weights]
    sample = window_returns[sample_columns].dropna()
    if len(sample) < 2:
        raise ValueError(
            f"fewer than two common return observations remain for the {window}-day window"
        )
    covariance, legacy_variance, legacy_cross_covariance = completion_covariances(
        sample, eligible, covariance_method, legacy_weights
    )
    legacy_net_weight = float(sum(legacy_weights.values()))
    new_money_weight = 1 - legacy_net_weight
    weights, solver_result = optimize_long_only(
        covariance,
        max_weight,
        annualization_days,
        legacy_cross_covariance,
        new_money_weight,
    )
    sleeve_variance = float(weights @ covariance @ weights)
    portfolio_variance = new_money_weight**2 * sleeve_variance
    if legacy_weights:
        assert legacy_variance is not None and legacy_cross_covariance is not None
        portfolio_variance += legacy_variance
        portfolio_variance += (
            2 * new_money_weight * float(legacy_cross_covariance @ weights)
        )
    annualized_volatility = float(
        np.sqrt(max(0.0, portfolio_variance) * annualization_days)
    )
    sleeve_annualized_volatility = float(
        np.sqrt(max(0.0, sleeve_variance) * annualization_days)
    )
    legacy_annualized_volatility = (
        float(
            np.sqrt(max(0.0, legacy_variance) * annualization_days)
            / abs(legacy_net_weight)
        )
        if legacy_variance is not None and legacy_net_weight != 0 else np.nan
    )
    fixed_contribution_volatility = (
        float(np.sqrt(max(0.0, legacy_variance) * annualization_days))
        if legacy_variance is not None else 0.0
    )
    lookthrough_available = len(legacy_weights) <= 1
    weight_by_symbol = dict(zip(eligible, weights))
    source_weights = pd.Series(
        {symbol: holdings_weights.get(symbol, np.nan) for symbol in stocks}, dtype=float
    )
    valid_source_weight = (
        source_weights.notna() & np.isfinite(source_weights) & (source_weights > 0)
    )
    total_source_weight = float(source_weights.where(valid_source_weight).sum())
    weight_rows = [
        {
            "window_days": window,
            "ticker": display_tickers[symbol],
            "yahoo_symbol": symbol,
            "sector": sectors.get(symbol, "Unclassified"),
            "weight": weight_by_symbol.get(symbol, 0.0),
            "portfolio_weight": new_money_weight * weight_by_symbol.get(symbol, 0.0),
            "etf_weight": (
                float(source_weights[symbol]) / total_source_weight
                if total_source_weight > 0 and bool(valid_source_weight[symbol]) else 0.0
            ),
            "total_lookthrough_weight": (
                new_money_weight * weight_by_symbol.get(symbol, 0.0)
                + legacy_net_weight
                * (
                    float(source_weights[symbol]) / total_source_weight
                    if total_source_weight > 0 and bool(valid_source_weight[symbol]) else 0.0
                )
            ) if lookthrough_available else np.nan,
            "included": symbol in weight_by_symbol,
            "available_observations": int(observation_counts.get(symbol, 0)),
        }
        for symbol in stocks
    ]
    weight_frame = pd.DataFrame(weight_rows).sort_values(
        ["window_days", "weight", "ticker"], ascending=[True, False, True]
    ).reset_index(drop=True)
    nonzero = int(np.sum(weights > 1e-8))
    summary = {
        "window_days": window,
        "portfolio_level": "stocks",
        "covariance_method": covariance_method,
        "requested_constituents": len(stocks),
        "eligible_constituents": len(eligible),
        "excluded_constituents": len(stocks) - len(eligible),
        "common_observations": len(sample),
        "coverage_required": coverage,
        "max_weight": max_weight,
        "annualization_days": annualization_days,
        "annualized_volatility": annualized_volatility,
        "new_money_sleeve_annualized_volatility": sleeve_annualized_volatility,
        "legacy_symbol": ",".join(legacy_weights),
        "legacy_weight": legacy_net_weight,
        "legacy_net_weight": legacy_net_weight,
        "legacy_gross_weight": float(sum(abs(weight) for weight in legacy_weights.values())),
        "legacy_long_weight": float(sum(max(weight, 0) for weight in legacy_weights.values())),
        "legacy_short_weight": float(sum(min(weight, 0) for weight in legacy_weights.values())),
        "new_money_weight": new_money_weight,
        "legacy_annualized_volatility": legacy_annualized_volatility,
        "fixed_positions_contribution_volatility": fixed_contribution_volatility,
        "lookthrough_available": lookthrough_available,
        "volatility_reduction_vs_legacy": (
            legacy_annualized_volatility - annualized_volatility
            if legacy_weights and np.isfinite(legacy_annualized_volatility) else np.nan
        ),
        "nonzero_positions": nonzero,
        "effective_positions": float(1 / np.sum(weights**2)),
        "largest_weight": float(weights.max()),
        "solver_iterations": int(getattr(solver_result, "nit", 0)),
        "solver_success": True,
    }
    return weight_frame, summary


def optimize_sector_window(
    returns: pd.DataFrame,
    stocks: list[str],
    display_tickers: dict[str, str],
    sectors: dict[str, str],
    holdings_weights: dict[str, float],
    window: int,
    covariance_method: str,
    max_sector_weight: float,
    coverage: float,
    annualization_days: float,
    legacy_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Optimize sector sleeves whose internal stock weights are fixed by holdings."""
    legacy_weights = legacy_weights or {}
    overlaps = sorted(set(legacy_weights) & set(stocks))
    if overlaps:
        raise ValueError(
            "legacy positions cannot also be candidate constituents: "
            + ", ".join(overlaps)
        )
    data_symbols = [*stocks, *legacy_weights]
    window_returns = returns.reindex(columns=data_symbols).dropna(how="all").tail(window)
    required_observations = max(2, math.ceil(window * coverage))
    observation_counts = window_returns.notna().sum()
    source_weights = pd.Series(
        {symbol: holdings_weights.get(symbol, np.nan) for symbol in stocks},
        dtype=float,
    )
    valid_weight = source_weights.notna() & np.isfinite(source_weights) & (source_weights > 0)
    total_source_weight = float(source_weights.where(valid_weight).sum())
    if total_source_weight <= 0:
        raise ValueError("sector optimization requires positive source holdings weights")
    eligible = [
        symbol
        for symbol in stocks
        if observation_counts.get(symbol, 0) >= required_observations
        and bool(valid_weight.get(symbol, False))
    ]
    if not eligible:
        raise ValueError(
            f"no positively weighted constituents meet {coverage:.1%} coverage "
            f"for the {window}-day sector window"
        )

    members_by_sector: dict[str, list[str]] = {}
    for symbol in eligible:
        members_by_sector.setdefault(sectors.get(symbol, "Unclassified"), []).append(symbol)

    within_sector_weights: dict[str, float] = {}
    sector_returns: dict[str, pd.Series] = {}
    for sector, members in members_by_sector.items():
        member_source_weights = source_weights[members]
        normalized = member_source_weights / member_source_weights.sum()
        within_sector_weights.update(normalized.to_dict())
        weighted_returns = window_returns[members].mul(normalized, axis=1)
        sector_returns[sector] = weighted_returns.sum(axis=1, min_count=len(members))

    sector_columns = list(sector_returns)
    sector_matrix = pd.DataFrame(sector_returns)
    for legacy_symbol in legacy_weights:
        sector_matrix[legacy_symbol] = window_returns[legacy_symbol]
    sector_sample = sector_matrix.dropna()
    if len(sector_sample) < 2:
        raise ValueError(
            f"fewer than two common sector-return observations remain for the "
            f"{window}-day window"
        )
    covariance, legacy_variance, legacy_cross_covariance = completion_covariances(
        sector_sample, sector_columns, covariance_method, legacy_weights
    )
    legacy_net_weight = float(sum(legacy_weights.values()))
    new_money_weight = 1 - legacy_net_weight
    sector_weights, solver_result = optimize_long_only(
        covariance,
        max_sector_weight,
        annualization_days,
        legacy_cross_covariance,
        new_money_weight,
    )
    sector_weight_map = dict(zip(sector_columns, sector_weights))
    sleeve_variance = float(sector_weights @ covariance @ sector_weights)
    portfolio_variance = new_money_weight**2 * sleeve_variance
    if legacy_weights:
        assert legacy_variance is not None and legacy_cross_covariance is not None
        portfolio_variance += legacy_variance
        portfolio_variance += (
            2 * new_money_weight * float(legacy_cross_covariance @ sector_weights)
        )
    annualized_volatility = float(
        np.sqrt(max(0.0, portfolio_variance) * annualization_days)
    )
    sleeve_annualized_volatility = float(
        np.sqrt(max(0.0, sleeve_variance) * annualization_days)
    )
    legacy_annualized_volatility = (
        float(
            np.sqrt(max(0.0, legacy_variance) * annualization_days)
            / abs(legacy_net_weight)
        )
        if legacy_variance is not None and legacy_net_weight != 0 else np.nan
    )
    fixed_contribution_volatility = (
        float(np.sqrt(max(0.0, legacy_variance) * annualization_days))
        if legacy_variance is not None else 0.0
    )
    lookthrough_available = len(legacy_weights) <= 1

    benchmark_sectors = {
        sectors.get(symbol, "Unclassified")
        for symbol in stocks
        if bool(valid_weight.get(symbol, False))
    }
    all_sectors = sorted(benchmark_sectors | set(sector_columns))
    sector_rows = []
    for sector in all_sectors:
        all_sector_symbols = [
            symbol
            for symbol in stocks
            if sectors.get(symbol, "Unclassified") == sector
            and bool(valid_weight.get(symbol, False))
        ]
        eligible_sector_symbols = members_by_sector.get(sector, [])
        original_sector_weight = float(source_weights[all_sector_symbols].sum())
        retained_sector_weight = float(source_weights[eligible_sector_symbols].sum())
        optimized_weight = sector_weight_map.get(sector, 0.0)
        etf_weight = original_sector_weight / total_source_weight
        sector_rows.append(
            {
                "window_days": window,
                "sector": sector,
                "weight": optimized_weight,
                "etf_weight": etf_weight,
                "active_weight": optimized_weight - etf_weight,
                "portfolio_weight": new_money_weight * optimized_weight,
                "legacy_lookthrough_weight": (
                    legacy_net_weight * etf_weight if lookthrough_available else np.nan
                ),
                "total_lookthrough_weight": (
                    new_money_weight * optimized_weight
                    + legacy_net_weight * etf_weight
                ) if lookthrough_available else np.nan,
                "constituents": len(eligible_sector_symbols),
                "original_holdings_weight": original_sector_weight,
                "retained_holdings_weight_fraction": (
                    retained_sector_weight / original_sector_weight
                    if original_sector_weight > 0 else np.nan
                ),
            }
        )
    sector_frame = pd.DataFrame(sector_rows).sort_values(
        ["window_days", "weight", "sector"], ascending=[True, False, True]
    ).reset_index(drop=True)

    constituent_rows = []
    for symbol in stocks:
        sector = sectors.get(symbol, "Unclassified")
        within_weight = within_sector_weights.get(symbol, 0.0)
        sector_weight = sector_weight_map.get(sector, 0.0)
        etf_weight = (
            float(source_weights[symbol]) / total_source_weight
            if bool(valid_weight.get(symbol, False)) else 0.0
        )
        optimized_weight = sector_weight * within_weight
        total_lookthrough_weight = (
            new_money_weight * optimized_weight
            + legacy_net_weight * etf_weight
        )
        constituent_rows.append(
            {
                "window_days": window,
                "ticker": display_tickers[symbol],
                "yahoo_symbol": symbol,
                "sector": sector,
                "weight": optimized_weight,
                "etf_weight": etf_weight,
                "active_weight": optimized_weight - etf_weight,
                "portfolio_weight": new_money_weight * optimized_weight,
                "legacy_lookthrough_weight": (
                    legacy_net_weight * etf_weight if lookthrough_available else np.nan
                ),
                "total_lookthrough_weight": (
                    total_lookthrough_weight if lookthrough_available else np.nan
                ),
                "sector_weight": sector_weight,
                "within_sector_weight": within_weight,
                "source_holdings_weight": source_weights.get(symbol, np.nan),
                "included": symbol in within_sector_weights,
                "available_observations": int(observation_counts.get(symbol, 0)),
            }
        )
    constituent_frame = pd.DataFrame(constituent_rows).sort_values(
        ["window_days", "weight", "ticker"], ascending=[True, False, True]
    ).reset_index(drop=True)

    retained_source_weight = float(source_weights[eligible].sum())
    summary = {
        "window_days": window,
        "portfolio_level": "sectors",
        "covariance_method": covariance_method,
        "requested_constituents": len(stocks),
        "eligible_constituents": len(eligible),
        "excluded_constituents": len(stocks) - len(eligible),
        "optimized_sectors": len(sector_columns),
        "common_observations": len(sector_sample),
        "coverage_required": coverage,
        "max_sector_weight": max_sector_weight,
        "annualization_days": annualization_days,
        "annualized_volatility": annualized_volatility,
        "new_money_sleeve_annualized_volatility": sleeve_annualized_volatility,
        "legacy_symbol": ",".join(legacy_weights),
        "legacy_weight": legacy_net_weight,
        "legacy_net_weight": legacy_net_weight,
        "legacy_gross_weight": float(sum(abs(weight) for weight in legacy_weights.values())),
        "legacy_long_weight": float(sum(max(weight, 0) for weight in legacy_weights.values())),
        "legacy_short_weight": float(sum(min(weight, 0) for weight in legacy_weights.values())),
        "new_money_weight": new_money_weight,
        "legacy_annualized_volatility": legacy_annualized_volatility,
        "fixed_positions_contribution_volatility": fixed_contribution_volatility,
        "lookthrough_available": lookthrough_available,
        "volatility_reduction_vs_legacy": (
            legacy_annualized_volatility - annualized_volatility
            if legacy_weights and np.isfinite(legacy_annualized_volatility) else np.nan
        ),
        "nonzero_sector_positions": int(np.sum(sector_weights > 1e-8)),
        "effective_sector_positions": float(1 / np.sum(sector_weights**2)),
        "largest_sector_weight": float(sector_weights.max()),
        "nonzero_constituent_positions": int(
            (constituent_frame["weight"] > 1e-8).sum()
        ),
        "retained_holdings_weight_fraction": (
            retained_source_weight / total_source_weight
            if total_source_weight > 0 else np.nan
        ),
        "etf_weight_total": float(sector_frame["etf_weight"].sum()),
        "active_weight_sum": float(sector_frame["active_weight"].sum()),
        "solver_iterations": int(getattr(solver_result, "nit", 0)),
        "solver_success": True,
    }
    return sector_frame, constituent_frame, summary


def print_completion_summary(summary: dict[str, object]) -> None:
    if not summary["legacy_symbol"]:
        return
    print(
        f"completion portfolio: fixed legacy net "
        f"{100 * summary['legacy_net_weight']:.2f}% "
        f"(gross {100 * summary['legacy_gross_weight']:.2f}%) + "
        f"{100 * summary['new_money_weight']:.2f}% new money; "
        f"fixed-position risk contribution: "
        f"{100 * summary['fixed_positions_contribution_volatility']:.2f}%; "
        f"new-money sleeve volatility: "
        f"{100 * summary['new_money_sleeve_annualized_volatility']:.2f}%"
    )
    legacy_volatility = float(summary["legacy_annualized_volatility"])
    if np.isfinite(legacy_volatility):
        print(
            f"normalized legacy-composition volatility: {100 * legacy_volatility:.2f}%; "
            f"completed volatility: {100 * summary['annualized_volatility']:.2f}%; "
            f"volatility reduction: "
            f"{100 * summary['volatility_reduction_vs_legacy']:.2f} pp"
        )


def main() -> int:
    overall_start = time.perf_counter()
    args = parse_args()
    if args.legacy_positions is not None:
        legacy_weights = read_legacy_positions(args.legacy_positions)
    elif args.legacy_symbol is not None:
        legacy_weights = {args.legacy_symbol: args.legacy_weight}
    else:
        legacy_weights = {}
    funds, stocks, shared_stem = prepare_holdings(args.holdings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices_path = (
        args.prices_file
        if args.prices_file is not None
        else args.output_dir / f"{shared_stem}_adjusted_close.csv"
    )

    load_start = time.perf_counter()
    download_events: list[tuple[Path, list[str]]] = []
    unavailable_downloads: list[str] = []
    missing_download_elapsed = 0.0
    if legacy_weights and args.legacy_prices_file is None:
        required_symbols = list(dict.fromkeys([*stocks, *legacy_weights]))
        all_prices, downloaded_symbols, unavailable, elapsed = ensure_saved_symbols(
            prices_path,
            required_symbols,
            max(args.days),
            args.code_dir,
            args.no_download_missing,
            list(legacy_weights),
        )
        prices = all_prices.reindex(columns=required_symbols)
        missing_download_elapsed += elapsed
        unavailable_downloads.extend(unavailable)
        if downloaded_symbols:
            download_events.append((prices_path, downloaded_symbols))
    else:
        all_prices, downloaded_symbols, unavailable, elapsed = ensure_saved_symbols(
            prices_path,
            stocks,
            max(args.days),
            args.code_dir,
            args.no_download_missing,
        )
        prices = all_prices.reindex(columns=stocks)
        missing_download_elapsed += elapsed
        unavailable_downloads.extend(unavailable)
        if downloaded_symbols:
            download_events.append((prices_path, downloaded_symbols))
        if legacy_weights:
            legacy_prices, downloaded_symbols, unavailable, elapsed = ensure_saved_symbols(
                args.legacy_prices_file,
                list(legacy_weights),
                max(args.days),
                args.code_dir,
                args.no_download_missing,
                list(legacy_weights),
            )
            missing_download_elapsed += elapsed
            unavailable_downloads.extend(unavailable)
            if downloaded_symbols:
                download_events.append(
                    (args.legacy_prices_file, downloaded_symbols)
                )
            prices = prices.join(
                legacy_prices[list(legacy_weights)], how="outer"
            ).sort_index()
    returns = daily_returns(prices)
    load_elapsed = time.perf_counter() - load_start

    calculation_start = time.perf_counter()
    results: list[dict[str, object]] = []
    for fund in funds:
        result: dict[str, object] = {"fund": fund}
        if args.portfolio_level in {"stocks", "both"}:
            weight_frames: list[pd.DataFrame] = []
            summaries: list[dict[str, object]] = []
            for window in args.days:
                weights, summary = optimize_fund_window(
                    returns,
                    fund.stocks,
                    fund.display_tickers,
                    fund.sectors,
                    fund.holdings_weights,
                    window,
                    args.covariance,
                    args.max_weight,
                    args.coverage,
                    args.annualization_days,
                    legacy_weights,
                )
                weight_frames.append(weights)
                summaries.append(summary)
            all_weights = pd.concat(weight_frames, ignore_index=True)
            all_summaries = pd.DataFrame(summaries)
            weights_path = args.output_dir / f"{fund.stem}_minimum_variance_weights.csv"
            summary_path = args.output_dir / f"{fund.stem}_minimum_variance_summary.csv"
            all_weights.to_csv(weights_path, index=False)
            all_summaries.to_csv(summary_path, index=False)
            result["stock_result"] = (
                all_weights, all_summaries, weights_path, summary_path
            )

        if args.portfolio_level in {"sectors", "both"}:
            sector_frames: list[pd.DataFrame] = []
            constituent_frames: list[pd.DataFrame] = []
            sector_summaries: list[dict[str, object]] = []
            for window in args.days:
                sector_weights, constituent_weights, summary = optimize_sector_window(
                    returns,
                    fund.stocks,
                    fund.display_tickers,
                    fund.sectors,
                    fund.holdings_weights,
                    window,
                    args.covariance,
                    args.max_sector_weight,
                    args.coverage,
                    args.annualization_days,
                    legacy_weights,
                )
                sector_frames.append(sector_weights)
                constituent_frames.append(constituent_weights)
                sector_summaries.append(summary)
            all_sector_weights = pd.concat(sector_frames, ignore_index=True)
            all_constituent_weights = pd.concat(constituent_frames, ignore_index=True)
            all_sector_summaries = pd.DataFrame(sector_summaries)
            sector_weights_path = (
                args.output_dir / f"{fund.stem}_sector_minimum_variance_weights.csv"
            )
            constituent_weights_path = (
                args.output_dir
                / f"{fund.stem}_sector_minimum_variance_constituent_weights.csv"
            )
            sector_summary_path = (
                args.output_dir / f"{fund.stem}_sector_minimum_variance_summary.csv"
            )
            all_sector_weights.to_csv(sector_weights_path, index=False)
            all_constituent_weights.to_csv(constituent_weights_path, index=False)
            all_sector_summaries.to_csv(sector_summary_path, index=False)
            result["sector_result"] = (
                all_sector_weights,
                all_constituent_weights,
                all_sector_summaries,
                sector_weights_path,
                constituent_weights_path,
                sector_summary_path,
            )
        results.append(result)
    calculation_elapsed = time.perf_counter() - calculation_start

    print(f"Using saved prices: {prices_path}")
    if args.legacy_prices_file is not None:
        print(f"Using legacy prices: {args.legacy_prices_file}")
    for updated_path, downloaded_symbols in download_events:
        print(
            f"Downloaded missing symbols into {updated_path}: "
            + ", ".join(downloaded_symbols)
        )
    if unavailable_downloads:
        print(
            "Downloaded symbols still unavailable and excluded: "
            + ", ".join(dict.fromkeys(unavailable_downloads))
        )
    if legacy_weights:
        legacy_net = sum(legacy_weights.values())
        legacy_gross = sum(abs(weight) for weight in legacy_weights.values())
        legacy_long = sum(max(weight, 0) for weight in legacy_weights.values())
        legacy_short = sum(min(weight, 0) for weight in legacy_weights.values())
        print("Fixed legacy positions:")
        legacy_display = pd.DataFrame(
            {
                "ticker": list(legacy_weights),
                "weight": [f"{100 * weight:+.2f}%" for weight in legacy_weights.values()],
            }
        )
        print(legacy_display.to_string(index=False))
        print(
            f"legacy net: {100 * legacy_net:.2f}%; gross: {100 * legacy_gross:.2f}%; "
            f"long: {100 * legacy_long:.2f}%; short: {100 * legacy_short:.2f}%; "
            f"new money: {100 * (1 - legacy_net):.2f}%"
        )
    for result in results:
        fund = result["fund"]
        print(f"\n=== {fund.path} ===")
        if "stock_result" in result:
            weights, summaries, weights_path, summary_path = result["stock_result"]
            print("\nStock-level optimization:")
            for summary in summaries.to_dict("records"):
                window = int(summary["window_days"])
                print(
                    f"{window}-day {summary['covariance_method']} covariance: "
                    f"{summary['eligible_constituents']}/{summary['requested_constituents']} "
                    f"constituents, {summary['common_observations']} common observations"
                )
                print(
                    f"annualized volatility: {100 * summary['annualized_volatility']:.2f}%; "
                    f"nonzero positions: {summary['nonzero_positions']}; "
                    f"effective positions: {summary['effective_positions']:.2f}; "
                    f"largest weight: {100 * summary['largest_weight']:.2f}%"
                )
                print_completion_summary(summary)
                top = weights[weights["window_days"] == window].head(10).copy()
                top["weight"] = (100 * top["weight"]).map(
                    lambda value: f"{value:.2f}%"
                )
                columns = ["ticker", "weight"]
                if summary["legacy_symbol"]:
                    top["portfolio_weight"] = (100 * top["portfolio_weight"]).map(
                        lambda value: f"{value:.2f}%"
                    )
                    top = top.rename(columns={"weight": "sleeve_weight"})
                    columns = ["ticker", "sleeve_weight", "portfolio_weight"]
                print("top weights:")
                print(top[columns].to_string(index=False))
            print(f"Wrote {weights_path}")
            print(f"Wrote {summary_path}")

        if "sector_result" in result:
            (
                sector_weights,
                constituent_weights,
                summaries,
                sector_weights_path,
                constituent_weights_path,
                sector_summary_path,
            ) = result["sector_result"]
            print("\nSector-sleeve optimization:")
            for summary in summaries.to_dict("records"):
                window = int(summary["window_days"])
                print(
                    f"{window}-day {summary['covariance_method']} covariance: "
                    f"{summary['optimized_sectors']} sectors, "
                    f"{summary['eligible_constituents']}/{summary['requested_constituents']} "
                    f"constituents, {summary['common_observations']} common observations"
                )
                print(
                    f"annualized volatility: {100 * summary['annualized_volatility']:.2f}%; "
                    f"nonzero sectors: {summary['nonzero_sector_positions']}; "
                    f"effective sectors: {summary['effective_sector_positions']:.2f}; "
                    f"largest sector: {100 * summary['largest_sector_weight']:.2f}%; "
                    f"holdings weight retained: "
                    f"{100 * summary['retained_holdings_weight_fraction']:.2f}%"
                )
                print_completion_summary(summary)
                top = sector_weights[
                    sector_weights["window_days"] == window
                ].copy()
                top["weight"] = (100 * top["weight"]).map(
                    lambda value: f"{value:.2f}%"
                )
                top["etf_weight"] = (100 * top["etf_weight"]).map(
                    lambda value: f"{value:.2f}%"
                )
                top["active_weight"] = (100 * top["active_weight"]).map(
                    lambda value: f"{value:+.2f}%"
                )
                top = top.rename(
                    columns={"weight": "min_vol", "active_weight": "difference"}
                )
                sector_columns_to_print = [
                    "sector", "min_vol", "etf_weight", "difference", "constituents"
                ]
                if summary["legacy_symbol"]:
                    top["portfolio_weight"] = (100 * top["portfolio_weight"]).map(
                        lambda value: f"{value:.2f}%"
                    )
                    sector_columns_to_print = [
                        "sector", "min_vol", "etf_weight", "difference",
                        "portfolio_weight", "constituents",
                    ]
                    if summary["lookthrough_available"]:
                        top["total_lookthrough_weight"] = (
                            100 * top["total_lookthrough_weight"]
                        ).map(lambda value: f"{value:.2f}%")
                        sector_columns_to_print.insert(
                            -1, "total_lookthrough_weight"
                        )
                print("sector weights:")
                print(
                    top[sector_columns_to_print].to_string(index=False)
                )
            print(f"Wrote {sector_weights_path}")
            print(f"Wrote {constituent_weights_path}")
            print(f"Wrote {sector_summary_path}")
    print(f"\nData load elapsed: {load_elapsed:.3f} seconds")
    if download_events:
        print(f"Missing-data download elapsed: {missing_download_elapsed:.3f} seconds")
    print(f"Calculations elapsed: {calculation_elapsed:.3f} seconds")
    print(f"Overall elapsed: {time.perf_counter() - overall_start:.3f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
