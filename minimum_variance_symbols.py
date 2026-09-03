"""Construct long-only minimum-variance portfolios from command-line symbols."""

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
    infer_price_file_symbols,
    read_symbols_file,
    symbol_stem,
    yahoo_symbol,
)
from portfolio_optimization import (
    candidate_benchmark_covariances,
    optimize_bounded,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a long-only minimum-variance portfolio of symbols."
    )
    parser.add_argument(
        "symbols", nargs="*",
        help=(
            "Yahoo symbols, e.g. IVV XLE XLP (optional with --symbols-file or "
            "--prices-file)"
        ),
    )
    parser.add_argument(
        "--symbols-file", type=Path, action="append", default=[], metavar="PATH",
        help=(
            "CSV with a symbol/ticker column or text file with one symbol per line; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--name", metavar="NAME",
        help="Short filename prefix for the default price cache and result CSVs",
    )
    parser.add_argument(
        "--max-symbols", "--max-sym", type=int, metavar="N",
        help="Use only the first N unique symbols after merging all inputs",
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
        help="Maximum weight per symbol, between 0 and 1 (default: 1)",
    )
    parser.add_argument(
        "--min-weight", "--min-wgt", "--min_wgt", type=float, default=0.0,
        help="Minimum weight per symbol; negative values allow shorting (default: 0)",
    )
    parser.add_argument(
        "--max-gross", type=float,
        help="Optional maximum gross exposure, sum(abs(weights)); must be at least 1",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.90,
        help="Minimum fraction of window returns required per symbol (default: 0.90)",
    )
    parser.add_argument(
        "--annualization-days", type=float, default=252.0,
        help="Trading days used to annualize volatility (default: 252)",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.04,
        help="Annual risk-free rate as a decimal for Sharpe ratios (default: 0.04)",
    )
    parser.add_argument(
        "--corrmat", action="store_true",
        help="Print and save each asset/portfolio correlation matrix",
    )
    parser.add_argument(
        "--benchmark",
        help="Yahoo symbol used as the tracking-error benchmark",
    )
    parser.add_argument(
        "--max-tracking-error", type=float,
        help="Maximum annualized tracking error as a decimal; requires --benchmark",
    )
    parser.add_argument(
        "--max-idio-vol", "--max-idiovol", type=float,
        help=(
            "Maximum annualized volatility after an optimal benchmark beta hedge; "
            "requires --benchmark"
        ),
    )
    parser.add_argument(
        "--prices-file", type=Path,
        help="Optional price-cache CSV; also supplies symbols when no other source is given",
    )
    parser.add_argument(
        "--no-download-missing", action="store_true",
        help="Fail rather than create/update the price cache from Yahoo",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Directory for the default price cache and results (default: output)",
    )
    parser.add_argument(
        "--code-dir", type=Path, default=DEFAULT_YFINANCE_CODE_DIR,
        help=r"Directory containing yfinance_util.py (default: C:\python\code)",
    )
    args = parser.parse_args()
    if args.benchmark is not None:
        args.benchmark = yahoo_symbol(args.benchmark)
    args.had_positional_symbols = bool(args.symbols)
    if args.name is not None:
        args.name = args.name.strip()
        invalid_name_characters = '<>:"/\\|?*'
        if (
            not args.name
            or args.name in {".", ".."}
            or args.name.endswith((" ", "."))
            or any(character in args.name for character in invalid_name_characters)
        ):
            parser.error(
                "--name must be a valid filename prefix without path separators"
            )
    combined_symbols = list(args.symbols)
    try:
        for path in args.symbols_file:
            combined_symbols.extend(read_symbols_file(path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    args.symbols = list(
        dict.fromkeys(
            yahoo_symbol(symbol)
            for symbol in combined_symbols
            if str(symbol).strip()
        )
    )
    args.symbols_inferred = False
    if not args.symbols:
        if args.prices_file is None:
            parser.error(
                "provide at least one positional symbol, --symbols-file, "
                "or --prices-file"
            )
        try:
            args.symbols = infer_price_file_symbols(args.prices_file)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            parser.error(str(exc))
        args.symbols_inferred = True
    if args.max_symbols is not None and args.max_symbols < 1:
        parser.error("--max-symbols must be at least 1")
    args.available_symbols = len(args.symbols)
    if args.max_symbols is not None:
        args.symbols = args.symbols[:args.max_symbols]
    args.days = list(dict.fromkeys(args.days))
    if any(window < 2 for window in args.days):
        parser.error("every --days value must be at least 2")
    if not math.isfinite(args.max_weight) or not 0 < args.max_weight <= 1:
        parser.error("--max-weight must be greater than 0 and at most 1")
    if not math.isfinite(args.min_weight) or args.min_weight > args.max_weight:
        parser.error("--min-weight must be finite and at most --max-weight")
    if args.max_gross is not None and (
        not math.isfinite(args.max_gross) or args.max_gross < 1
    ):
        parser.error("--max-gross must be finite and at least 1")
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be greater than 0 and at most 1")
    if args.annualization_days <= 0:
        parser.error("--annualization-days must be positive")
    if args.risk_free_rate <= -1:
        parser.error("--risk-free-rate must be greater than -1")
    if args.max_tracking_error is not None:
        if args.benchmark is None:
            parser.error("--max-tracking-error requires --benchmark")
        if (
            not math.isfinite(args.max_tracking_error)
            or args.max_tracking_error < 0
        ):
            parser.error("--max-tracking-error must be finite and nonnegative")
    if args.max_idio_vol is not None:
        if args.benchmark is None:
            parser.error("--max-idio-vol requires --benchmark")
        if not math.isfinite(args.max_idio_vol) or args.max_idio_vol < 0:
            parser.error("--max-idio-vol must be finite and nonnegative")
    return args


def output_stem(args: argparse.Namespace) -> str:
    """Choose a concise, deterministic prefix for cache and result files."""
    if args.name is not None:
        return args.name
    if args.symbols_inferred and args.prices_file is not None:
        stem = args.prices_file.stem
        suffix = "_adjusted_close"
        inferred_stem = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        return inferred_stem or symbol_stem(args.symbols)
    if len(args.symbols_file) == 1 and not args.had_positional_symbols:
        return args.symbols_file[0].stem
    return symbol_stem(args.symbols)


def optimize_window(
    returns: pd.DataFrame,
    symbols: list[str],
    window: int,
    covariance_method: str,
    min_weight: float,
    max_weight: float,
    max_gross: float | None,
    coverage: float,
    annualization_days: float,
    risk_free_rate: float,
    include_correlation_matrix: bool,
    benchmark: str | None,
    max_tracking_error: float | None,
    max_idio_vol: float | None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    analysis_symbols = list(symbols)
    if benchmark is not None and benchmark not in analysis_symbols:
        analysis_symbols.append(benchmark)
    window_returns = (
        returns.reindex(columns=analysis_symbols).dropna(how="all").tail(window)
    )
    required = max(2, math.ceil(window * coverage))
    observations = window_returns.notna().sum()
    eligible = [symbol for symbol in symbols if observations.get(symbol, 0) >= required]
    if not eligible:
        raise ValueError(
            f"no symbols meet {coverage:.1%} coverage for the {window}-day window"
        )
    if benchmark is not None and observations.get(benchmark, 0) < required:
        raise ValueError(
            f"benchmark {benchmark} does not meet {coverage:.1%} coverage for "
            f"the {window}-day window"
        )
    covariance_symbols = list(eligible)
    if benchmark is not None and benchmark not in covariance_symbols:
        covariance_symbols.append(benchmark)
    sample = window_returns[covariance_symbols].dropna()
    if len(sample) < 2:
        raise ValueError(
            f"fewer than two common observations remain for the {window}-day window"
        )
    candidate_sample = sample[eligible]
    covariance, benchmark_covariance, benchmark_variance = (
        candidate_benchmark_covariances(
            sample, eligible, covariance_method, benchmark
        )
    )
    weights, solver_result = optimize_bounded(
        covariance,
        min_weight,
        max_weight,
        annualization_days,
        max_gross=max_gross,
        benchmark_covariance=benchmark_covariance,
        benchmark_variance=benchmark_variance,
        max_tracking_error=max_tracking_error,
        max_idiosyncratic_volatility=max_idio_vol,
    )
    portfolio_variance = float(max(0.0, weights @ covariance @ weights))
    asset_variances = np.maximum(0.0, np.diag(covariance))
    asset_volatilities = np.sqrt(asset_variances * annualization_days)
    portfolio_volatility = float(np.sqrt(portfolio_variance * annualization_days))
    asset_portfolio_covariances = covariance @ weights
    correlation_denominators = np.sqrt(asset_variances * portfolio_variance)
    asset_correlations = np.divide(
        asset_portfolio_covariances,
        correlation_denominators,
        out=np.full(len(eligible), np.nan),
        where=correlation_denominators > 0,
    )
    asset_correlations = np.clip(asset_correlations, -1.0, 1.0)

    portfolio_daily_returns = candidate_sample.to_numpy() @ weights
    asset_period_returns = (1.0 + candidate_sample).prod(axis=0).to_numpy() - 1.0
    portfolio_period_return = float(np.prod(1.0 + portfolio_daily_returns) - 1.0)
    daily_risk_free_rate = math.expm1(
        math.log1p(risk_free_rate) / annualization_days
    )
    asset_excess_returns = (
        candidate_sample.mean(axis=0).to_numpy() - daily_risk_free_rate
    ) * annualization_days
    portfolio_excess_return = float(
        (portfolio_daily_returns.mean() - daily_risk_free_rate)
        * annualization_days
    )
    asset_sharpe_ratios = np.divide(
        asset_excess_returns,
        asset_volatilities,
        out=np.full(len(eligible), np.nan),
        where=asset_volatilities > 0,
    )
    portfolio_sharpe_ratio = (
        portfolio_excess_return / portfolio_volatility
        if portfolio_volatility > 0
        else float("nan")
    )

    benchmark_volatility = None
    benchmark_period_return = None
    active_period_return = None
    tracking_error = None
    information_ratio = None
    tracking_error_binding = None
    portfolio_beta = None
    idiosyncratic_volatility = None
    implied_benchmark_hedge = None
    hedged_gross_exposure = None
    hedged_net_exposure = None
    idiosyncratic_volatility_binding = None
    if benchmark is not None:
        benchmark_daily_returns = sample[benchmark].to_numpy()
        benchmark_volatility = float(
            np.sqrt(max(0.0, benchmark_variance) * annualization_days)
        )
        benchmark_period_return = float(
            np.prod(1.0 + benchmark_daily_returns) - 1.0
        )
        active_period_return = portfolio_period_return - benchmark_period_return
        active_variance = float(
            portfolio_variance
            - 2 * benchmark_covariance @ weights
            + benchmark_variance
        )
        tracking_error = float(
            np.sqrt(max(0.0, active_variance) * annualization_days)
        )
        annualized_active_return = float(
            (portfolio_daily_returns - benchmark_daily_returns).mean()
            * annualization_days
        )
        information_ratio = (
            annualized_active_return / tracking_error
            if tracking_error > 0
            else float("nan")
        )
        if max_tracking_error is not None:
            tracking_error_binding = bool(
                tracking_error >= max_tracking_error - 1e-6
            )
        if benchmark_variance > 0:
            portfolio_benchmark_covariance = float(benchmark_covariance @ weights)
            portfolio_beta = portfolio_benchmark_covariance / benchmark_variance
            residual_variance = float(
                portfolio_variance
                - portfolio_benchmark_covariance**2 / benchmark_variance
            )
            idiosyncratic_volatility = float(
                np.sqrt(max(0.0, residual_variance) * annualization_days)
            )
            implied_benchmark_hedge = -portfolio_beta
            hedged_weights = weights.copy()
            if benchmark in eligible:
                hedged_weights[eligible.index(benchmark)] += implied_benchmark_hedge
                hedged_gross_exposure = float(np.abs(hedged_weights).sum())
            else:
                hedged_gross_exposure = float(
                    np.abs(hedged_weights).sum() + abs(implied_benchmark_hedge)
                )
            hedged_net_exposure = float(weights.sum() + implied_benchmark_hedge)
            if max_idio_vol is not None:
                idiosyncratic_volatility_binding = bool(
                    idiosyncratic_volatility >= max_idio_vol - 1e-6
                )

    correlation_matrix = None
    if include_correlation_matrix:
        asset_correlation_denominators = np.sqrt(
            np.outer(asset_variances, asset_variances)
        )
        asset_correlation_matrix = np.divide(
            covariance,
            asset_correlation_denominators,
            out=np.full_like(covariance, np.nan),
            where=asset_correlation_denominators > 0,
        )
        asset_correlation_matrix = np.clip(asset_correlation_matrix, -1.0, 1.0)
        positive_variance = asset_variances > 0
        diagonal = np.diag_indices_from(asset_correlation_matrix)
        asset_correlation_matrix[diagonal] = np.where(
            positive_variance, 1.0, np.nan
        )
        augmented = np.full((len(eligible) + 1, len(eligible) + 1), np.nan)
        augmented[:-1, :-1] = asset_correlation_matrix
        augmented[:-1, -1] = asset_correlations
        augmented[-1, :-1] = asset_correlations
        if portfolio_variance > 0:
            augmented[-1, -1] = 1.0
        labels = [*eligible, "*MIN_VOL*"]
        correlation_matrix = pd.DataFrame(augmented, index=labels, columns=labels)

    weight_map = dict(zip(eligible, weights))
    volatility_map = dict(zip(eligible, asset_volatilities))
    correlation_map = dict(zip(eligible, asset_correlations))
    return_map = dict(zip(eligible, asset_period_returns))
    sharpe_map = dict(zip(eligible, asset_sharpe_ratios))
    frame = pd.DataFrame(
        {
            "window_days": window,
            "symbol": symbols,
            "weight": [weight_map.get(symbol, 0.0) for symbol in symbols],
            "annualized_volatility": [
                volatility_map.get(symbol, np.nan) for symbol in symbols
            ],
            "correlation_to_portfolio": [
                correlation_map.get(symbol, np.nan) for symbol in symbols
            ],
            "period_return": [return_map.get(symbol, np.nan) for symbol in symbols],
            "sharpe_ratio": [sharpe_map.get(symbol, np.nan) for symbol in symbols],
            "included": [symbol in weight_map for symbol in symbols],
            "available_observations": [int(observations.get(symbol, 0)) for symbol in symbols],
        }
    ).sort_values(["weight", "symbol"], ascending=[False, True]).reset_index(drop=True)
    summary = {
        "window_days": window,
        "covariance_method": covariance_method,
        "requested_symbols": len(symbols),
        "eligible_symbols": len(eligible),
        "excluded_symbols": len(symbols) - len(eligible),
        "common_observations": len(sample),
        "coverage_required": coverage,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "max_gross": max_gross,
        "annualization_days": annualization_days,
        "risk_free_rate": risk_free_rate,
        "benchmark": benchmark,
        "max_tracking_error": max_tracking_error,
        "benchmark_annualized_volatility": benchmark_volatility,
        "benchmark_period_return": benchmark_period_return,
        "active_period_return": active_period_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "tracking_error_binding": tracking_error_binding,
        "max_idiosyncratic_volatility": max_idio_vol,
        "portfolio_beta": portfolio_beta,
        "idiosyncratic_volatility": idiosyncratic_volatility,
        "implied_benchmark_hedge": implied_benchmark_hedge,
        "hedged_gross_exposure": hedged_gross_exposure,
        "hedged_net_exposure": hedged_net_exposure,
        "idiosyncratic_volatility_binding": idiosyncratic_volatility_binding,
        "annualized_volatility": portfolio_volatility,
        "period_return": portfolio_period_return,
        "sharpe_ratio": portfolio_sharpe_ratio,
        "nonzero_positions": int(np.sum(np.abs(weights) > 1e-8)),
        "effective_positions": float(1 / np.sum(weights**2)),
        "largest_weight": float(weights.max()),
        "gross_exposure": float(np.abs(weights).sum()),
        "long_exposure": float(weights[weights > 0].sum()),
        "short_exposure": float(-weights[weights < 0].sum()),
        "largest_long": float(max(0.0, weights.max())),
        "largest_short": float(min(0.0, weights.min())),
        "solver_iterations": int(getattr(solver_result, "nit", 0)),
        "solver_success": True,
    }
    return frame, summary, correlation_matrix


def main() -> int:
    overall_start = time.perf_counter()
    args = parse_args()
    if args.symbols_inferred:
        print(
            f"Inferred {args.available_symbols} unique symbols from "
            f"{args.prices_file}"
        )
    if len(args.symbols) < args.available_symbols:
        print(
            f"Using first {len(args.symbols)} of "
            f"{args.available_symbols} unique symbols"
        )
    if args.min_weight < 0 and args.max_gross is None:
        print(
            "Warning: negative weights are enabled without a gross-exposure limit."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    prices_path = (
        args.prices_file
        if args.prices_file is not None
        else args.output_dir / f"{stem}_adjusted_close.csv"
    )

    data_start = time.perf_counter()
    data_symbols = list(args.symbols)
    if args.benchmark is not None and args.benchmark not in data_symbols:
        data_symbols.append(args.benchmark)
    prices, downloaded, unavailable, download_elapsed = ensure_saved_symbols(
        prices_path,
        data_symbols,
        max(args.days),
        args.code_dir,
        args.no_download_missing,
        critical_symbols=[args.benchmark] if args.benchmark is not None else None,
    )
    prices = prices.reindex(columns=data_symbols)
    returns = daily_returns(prices)
    data_elapsed = time.perf_counter() - data_start

    calculation_start = time.perf_counter()
    weight_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    correlation_matrices: dict[int, pd.DataFrame] = {}
    for window in args.days:
        weights, summary, correlation_matrix = optimize_window(
            returns,
            args.symbols,
            window,
            args.covariance,
            args.min_weight,
            args.max_weight,
            args.max_gross,
            args.coverage,
            args.annualization_days,
            args.risk_free_rate,
            args.corrmat,
            args.benchmark,
            args.max_tracking_error,
            args.max_idio_vol,
        )
        weight_frames.append(weights)
        summaries.append(summary)
        if correlation_matrix is not None:
            correlation_matrices[window] = correlation_matrix
    all_weights = pd.concat(weight_frames, ignore_index=True)
    all_summaries = pd.DataFrame(summaries)
    weights_path = args.output_dir / f"{stem}_minimum_variance_weights.csv"
    summary_path = args.output_dir / f"{stem}_minimum_variance_summary.csv"
    all_weights.to_csv(weights_path, index=False)
    all_summaries.to_csv(summary_path, index=False)
    correlation_paths: dict[int, Path] = {}
    for window, correlation_matrix in correlation_matrices.items():
        path = args.output_dir / f"{stem}_{window}d_correlation_matrix.csv"
        correlation_matrix.to_csv(path, index_label="symbol")
        correlation_paths[window] = path
    calculation_elapsed = time.perf_counter() - calculation_start

    print(f"Using prices: {prices_path}")
    if downloaded:
        print("Downloaded symbols: " + ", ".join(downloaded))
    if unavailable:
        print("Unavailable symbols excluded: " + ", ".join(unavailable))
    for summary in summaries:
        window = int(summary["window_days"])
        print(
            f"\n{window}-day {summary['covariance_method']} covariance: "
            f"{summary['eligible_symbols']}/{summary['requested_symbols']} symbols, "
            f"{summary['common_observations']} common observations"
        )
        print(
            f"annualized volatility: {100 * summary['annualized_volatility']:.2f}%; "
            f"period return: {100 * summary['period_return']:.2f}%; "
            f"Sharpe ratio: {summary['sharpe_ratio']:.3f} "
            f"(risk-free rate: {100 * summary['risk_free_rate']:.2f}%)"
        )
        if summary["benchmark"] is not None:
            constraint_text = ""
            if summary["max_tracking_error"] is not None:
                binding = "yes" if summary["tracking_error_binding"] else "no"
                constraint_text = (
                    f"; limit: {100 * summary['max_tracking_error']:.2f}%"
                    f"; binding: {binding}"
                )
            print(
                f"benchmark {summary['benchmark']}: volatility: "
                f"{100 * summary['benchmark_annualized_volatility']:.2f}%; "
                f"period return: {100 * summary['benchmark_period_return']:.2f}%"
            )
            print(
                f"tracking error: {100 * summary['tracking_error']:.2f}%"
                f"{constraint_text}; active return: "
                f"{100 * summary['active_period_return']:.2f}%; "
                f"information ratio: {summary['information_ratio']:.3f}"
            )
            if summary["idiosyncratic_volatility"] is not None:
                idio_constraint_text = ""
                if summary["max_idiosyncratic_volatility"] is not None:
                    binding = (
                        "yes"
                        if summary["idiosyncratic_volatility_binding"]
                        else "no"
                    )
                    idio_constraint_text = (
                        "; limit: "
                        f"{100 * summary['max_idiosyncratic_volatility']:.2f}%"
                        f"; binding: {binding}"
                    )
                print(
                    f"portfolio beta: {summary['portfolio_beta']:.3f}; "
                    f"idiosyncratic volatility: "
                    f"{100 * summary['idiosyncratic_volatility']:.2f}%"
                    f"{idio_constraint_text}; implied {summary['benchmark']} hedge: "
                    f"{100 * summary['implied_benchmark_hedge']:.2f}%; "
                    f"hedged net exposure: "
                    f"{100 * summary['hedged_net_exposure']:.2f}%; "
                    f"hedged gross exposure: "
                    f"{100 * summary['hedged_gross_exposure']:.2f}%"
                )
        print(
            f"nonzero positions: {summary['nonzero_positions']}; "
            f"effective positions: {summary['effective_positions']:.2f}; "
            f"largest weight: {100 * summary['largest_weight']:.2f}%"
        )
        if summary["min_weight"] < 0 or summary["max_gross"] is not None:
            print(
                f"long exposure: {100 * summary['long_exposure']:.2f}%; "
                f"short exposure: {100 * summary['short_exposure']:.2f}%; "
                f"gross exposure: {100 * summary['gross_exposure']:.2f}%; "
                f"largest long: {100 * summary['largest_long']:.2f}%; "
                f"largest short: {100 * summary['largest_short']:.2f}%"
            )
        display = all_weights[all_weights["window_days"] == window].copy()
        display["weight"] = (100 * display["weight"]).map(lambda value: f"{value:.2f}%")
        for column in ["annualized_volatility", "period_return"]:
            display[column] = display[column].map(
                lambda value: f"{100 * value:.2f}%" if pd.notna(value) else ""
            )
        for column in ["correlation_to_portfolio", "sharpe_ratio"]:
            display[column] = display[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else ""
            )
        display = display.rename(
            columns={
                "annualized_volatility": "volatility",
                "correlation_to_portfolio": "corr_portfolio",
                "period_return": "return",
                "sharpe_ratio": "sharpe",
            }
        )
        portfolio_row = pd.DataFrame(
            [
                {
                    "symbol": "*MIN_VOL*",
                    "weight": "100.00%",
                    "volatility": f"{100 * summary['annualized_volatility']:.2f}%",
                    "corr_portfolio": "1.000",
                    "return": f"{100 * summary['period_return']:.2f}%",
                    "sharpe": f"{summary['sharpe_ratio']:.3f}",
                }
            ]
        )
        display = pd.concat([display, portfolio_row], ignore_index=True)
        print(
            display[
                ["symbol", "weight", "volatility", "corr_portfolio", "return", "sharpe"]
            ].to_string(index=False)
        )
        if window in correlation_matrices:
            print("\ncorrelation matrix:")
            print(
                correlation_matrices[window].to_string(
                    float_format=lambda value: f"{value:.3f}"
                )
            )
    print(f"\nWrote {weights_path}")
    print(f"Wrote {summary_path}")
    for path in correlation_paths.values():
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
