"""Shared walk-forward engine for minimum- and mean-variance portfolios."""

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
    ensure_saved_history,
    infer_price_file_symbols,
    read_symbols_file,
    symbol_stem,
    yahoo_symbol,
)
from portfolio_optimization import (
    candidate_benchmark_covariances,
    optimize_bounded,
)


def parse_args(strategy: str) -> argparse.Namespace:
    strategy_label = strategy.replace("-", " ")
    parser = argparse.ArgumentParser(
        description=f"Walk-forward backtest of a {strategy_label} portfolio."
    )
    parser.add_argument("symbols", nargs="*", help="Investable Yahoo symbols")
    parser.add_argument(
        "--symbols-file", type=Path, action="append", default=[], metavar="PATH",
        help="CSV symbol/ticker column or one-symbol-per-line text file; repeatable",
    )
    parser.add_argument("--name", help="Short filename prefix for cache and outputs")
    parser.add_argument(
        "--max-symbols", "--max-sym", type=int, metavar="N",
        help="Use only the first N unique investable symbols",
    )
    parser.add_argument(
        "--min-symbols", "--min-sym", type=int, metavar="N",
        help="Start only when at least N symbols meet lookback coverage",
    )
    parser.add_argument(
        "--lookback", type=int, default=126,
        help="Return observations used at each rebalance (default: 126)",
    )
    parser.add_argument(
        "--rebalance-days", type=int, default=63,
        help="Trading days between rebalances (default: 63)",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=1260,
        help="Requested out-of-sample trading days (default: 1260)",
    )
    parser.add_argument(
        "--covariance", choices=["ledoit-wolf", "sample"], default="ledoit-wolf",
        help="Covariance estimator (default: ledoit-wolf)",
    )
    if strategy == "mean-variance":
        parser.add_argument(
            "--return-lookback", type=int,
            help="Return observations used to estimate expected returns (default: --lookback)",
        )
        parser.add_argument(
            "--return-estimator",
            choices=["historical", "ewma", "shrinkage"],
            default="shrinkage",
            help="Expected-return estimator (default: shrinkage)",
        )
        parser.add_argument(
            "--mean-shrinkage", type=float, default=0.50,
            help="Fraction shrunk toward the cross-sectional mean (default: 0.50)",
        )
        parser.add_argument(
            "--ewma-halflife", type=float, default=63.0,
            help="EWMA half-life in trading days (default: 63)",
        )
        mean_variance_group = parser.add_mutually_exclusive_group()
        mean_variance_group.add_argument(
            "--risk-aversion", type=float,
            help="Annual mean-variance risk aversion (default: 3)",
        )
        mean_variance_group.add_argument(
            "--target-return", type=float,
            help="Minimum annualized expected portfolio return",
        )
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument(
        "--min-weight", "--min-wgt", "--min_wgt", type=float, default=0.0
    )
    parser.add_argument("--max-gross", type=float)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--annualization-days", type=float, default=252.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--benchmark", help="Yahoo benchmark symbol")
    parser.add_argument(
        "--vol-ratio-analysis", action="store_true",
        help=(
            "Analyze subsequent performance by ex-ante portfolio/benchmark "
            "volatility ratio"
        ),
    )
    parser.add_argument(
        "--plot",
        action="append",
        choices=["performance", "volatility", "vol-ratio", "weights", "all"],
        default=[],
        help="Save a plot; repeat for multiple plots or use 'all'",
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Display requested plots interactively after saving them",
    )
    parser.add_argument(
        "--plot-format", choices=["png", "pdf", "svg"], default="png",
        help="Saved plot format (default: png)",
    )
    parser.add_argument(
        "--plot-top-weights", type=int, default=10, metavar="N",
        help="Show the N largest average absolute weights separately (default: 10)",
    )
    parser.add_argument("--max-tracking-error", type=float)
    parser.add_argument("--max-idio-vol", "--max-idiovol", type=float)
    parser.add_argument(
        "--transaction-cost-bps", type=float, default=0.0,
        help="One-way trading cost in basis points (default: 0)",
    )
    parser.add_argument(
        "--missing-return", choices=["zero", "error"], default="zero",
        help="Treatment of missing held-asset/benchmark returns (default: zero)",
    )
    parser.add_argument("--prices-file", type=Path)
    parser.add_argument("--no-download-missing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_YFINANCE_CODE_DIR)
    args = parser.parse_args()
    args.strategy = strategy
    if strategy == "mean-variance":
        if args.return_lookback is None:
            args.return_lookback = args.lookback
        if args.risk_aversion is None and args.target_return is None:
            args.risk_aversion = 3.0
    else:
        args.return_lookback = None
        args.return_estimator = None
        args.mean_shrinkage = None
        args.ewma_halflife = None
        args.risk_aversion = None
        args.target_return = None

    args.had_positional_symbols = bool(args.symbols)
    combined = list(args.symbols)
    try:
        for path in args.symbols_file:
            combined.extend(read_symbols_file(path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    args.symbols = list(
        dict.fromkeys(
            yahoo_symbol(symbol) for symbol in combined if str(symbol).strip()
        )
    )
    args.symbols_inferred = False
    if not args.symbols:
        if args.prices_file is None:
            parser.error(
                "provide positional symbols, --symbols-file, or --prices-file"
            )
        try:
            args.symbols = infer_price_file_symbols(args.prices_file)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            parser.error(str(exc))
        args.symbols_inferred = True
    if args.max_symbols is not None:
        if args.max_symbols < 1:
            parser.error("--max-symbols must be at least 1")
        args.symbols = args.symbols[:args.max_symbols]
    if args.min_symbols is not None:
        if args.min_symbols < 1:
            parser.error("--min-symbols must be at least 1")
        if args.min_symbols > len(args.symbols):
            parser.error(
                f"--min-symbols cannot exceed the {len(args.symbols)} "
                "investable symbols"
            )
    if args.benchmark is not None:
        args.benchmark = yahoo_symbol(args.benchmark)

    if args.name is not None:
        args.name = args.name.strip()
        if (
            not args.name
            or args.name in {".", ".."}
            or args.name.endswith((" ", "."))
            or any(character in args.name for character in '<>:"/\\|?*')
        ):
            parser.error("--name must be a valid filename prefix")
    for option, value in [
        ("--lookback", args.lookback),
        ("--rebalance-days", args.rebalance_days),
        ("--backtest-days", args.backtest_days),
    ]:
        if value < 2:
            parser.error(f"{option} must be at least 2")
    if args.return_lookback is not None and args.return_lookback < 2:
        parser.error("--return-lookback must be at least 2")
    if args.risk_aversion is not None and (
        not math.isfinite(args.risk_aversion) or args.risk_aversion <= 0
    ):
        parser.error("--risk-aversion must be finite and positive")
    if args.target_return is not None and not math.isfinite(args.target_return):
        parser.error("--target-return must be finite")
    if args.mean_shrinkage is not None and not 0 <= args.mean_shrinkage <= 1:
        parser.error("--mean-shrinkage must be between 0 and 1")
    if args.ewma_halflife is not None and (
        not math.isfinite(args.ewma_halflife) or args.ewma_halflife <= 0
    ):
        parser.error("--ewma-halflife must be finite and positive")
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
    if not math.isfinite(args.risk_free_rate) or args.risk_free_rate <= -1:
        parser.error("--risk-free-rate must be finite and greater than -1")
    if args.transaction_cost_bps < 0 or not math.isfinite(args.transaction_cost_bps):
        parser.error("--transaction-cost-bps must be finite and nonnegative")
    if args.vol_ratio_analysis and args.benchmark is None:
        parser.error("--vol-ratio-analysis requires --benchmark")
    if "all" in args.plot:
        args.plot = ["performance", "volatility", "vol-ratio", "weights"]
    else:
        args.plot = list(dict.fromkeys(args.plot))
    if args.show_plot and not args.plot:
        parser.error("--show-plot requires at least one --plot")
    benchmark_plots = {"volatility", "vol-ratio"}.intersection(args.plot)
    if benchmark_plots and args.benchmark is None:
        parser.error(
            "--plot " + ", ".join(sorted(benchmark_plots))
            + " requires --benchmark"
        )
    if args.plot_top_weights < 1:
        parser.error("--plot-top-weights must be at least 1")
    for option, value in [
        ("--max-tracking-error", args.max_tracking_error),
        ("--max-idio-vol", args.max_idio_vol),
    ]:
        if value is not None:
            if args.benchmark is None:
                parser.error(f"{option} requires --benchmark")
            if value < 0 or not math.isfinite(value):
                parser.error(f"{option} must be finite and nonnegative")
    return args


def output_stem(args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    if args.symbols_inferred and args.prices_file is not None:
        stem = args.prices_file.stem
        suffix = "_adjusted_close"
        inferred = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        return inferred or symbol_stem(args.symbols)
    if len(args.symbols_file) == 1 and not args.had_positional_symbols:
        return args.symbols_file[0].stem
    return symbol_stem(args.symbols)


def eligible_training_data(
    training: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    covariance_training = training.iloc[-args.lookback:]
    covariance_required = max(2, math.ceil(args.lookback * args.coverage))
    covariance_observations = covariance_training.notna().sum()
    eligible = [
        symbol for symbol in args.symbols
        if covariance_observations.get(symbol, 0) >= covariance_required
    ]
    return_training = None
    if args.strategy == "mean-variance":
        return_training = training.iloc[-args.return_lookback:]
        return_required = max(
            2, math.ceil(args.return_lookback * args.coverage)
        )
        return_observations = return_training.notna().sum()
        eligible = [
            symbol for symbol in eligible
            if return_observations.get(symbol, 0) >= return_required
        ]
    return covariance_training, return_training, eligible


def estimate_expected_returns(
    training: pd.DataFrame,
    eligible: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    selected = training[eligible].dropna()
    if len(selected) < 2:
        raise ValueError(
            "fewer than two common expected-return observations remain"
        )
    if args.return_estimator == "ewma":
        daily_means = selected.ewm(
            halflife=args.ewma_halflife,
            adjust=True,
        ).mean().iloc[-1]
    else:
        daily_means = selected.mean()
        if args.return_estimator == "shrinkage":
            center = float(daily_means.mean())
            daily_means = (
                (1.0 - args.mean_shrinkage) * daily_means
                + args.mean_shrinkage * center
            )
    expected_returns = daily_means.to_numpy(dtype=float) * args.annualization_days
    if not np.isfinite(expected_returns).all():
        raise ValueError("expected-return estimation produced non-finite values")
    return expected_returns, len(selected)


def solve_rebalance(
    training: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], dict[str, object], dict[str, float]]:
    covariance_training, return_training, eligible = eligible_training_data(
        training, args
    )
    required = max(2, math.ceil(args.lookback * args.coverage))
    observations = covariance_training.notna().sum()
    if not eligible:
        raise ValueError("no investable symbols meet the coverage requirement")
    if (
        args.benchmark is not None
        and observations.get(args.benchmark, 0) < required
    ):
        raise ValueError(
            f"benchmark {args.benchmark} does not meet the coverage requirement"
        )
    covariance_symbols = list(eligible)
    if args.benchmark is not None and args.benchmark not in covariance_symbols:
        covariance_symbols.append(args.benchmark)
    sample = covariance_training[covariance_symbols].dropna()
    if len(sample) < 2:
        raise ValueError("fewer than two common training observations remain")
    covariance, benchmark_covariance, benchmark_variance = (
        candidate_benchmark_covariances(
            sample, eligible, args.covariance, args.benchmark
        )
    )
    expected_returns = None
    expected_return_observations = None
    expected_return_map: dict[str, float] = {}
    if args.strategy == "mean-variance":
        expected_returns, expected_return_observations = estimate_expected_returns(
            return_training, eligible, args
        )
        expected_return_map = dict(zip(eligible, expected_returns))
    eligible_weights, result = optimize_bounded(
        covariance,
        args.min_weight,
        args.max_weight,
        args.annualization_days,
        max_gross=args.max_gross,
        benchmark_covariance=benchmark_covariance,
        benchmark_variance=benchmark_variance,
        max_tracking_error=args.max_tracking_error,
        max_idiosyncratic_volatility=args.max_idio_vol,
        expected_returns=expected_returns,
        risk_aversion=args.risk_aversion,
        target_return=args.target_return,
    )
    weight_map = dict(zip(eligible, eligible_weights))
    weights = np.array([weight_map.get(symbol, 0.0) for symbol in args.symbols])
    portfolio_variance = float(
        max(0.0, eligible_weights @ covariance @ eligible_weights)
    )
    diagnostics: dict[str, object] = {
        "eligible_symbols": len(eligible),
        "common_observations": len(sample),
        "expected_annualized_volatility": math.sqrt(
            portfolio_variance * args.annualization_days
        ),
        "gross_exposure": float(np.abs(eligible_weights).sum()),
        "solver_iterations": int(getattr(result, "nit", 0)),
    }
    if expected_returns is not None:
        diagnostics["expected_return_observations"] = expected_return_observations
        diagnostics["expected_portfolio_return"] = float(
            expected_returns @ eligible_weights
        )
    if args.benchmark is not None:
        benchmark_annualized_volatility = math.sqrt(
            benchmark_variance * args.annualization_days
        )
        portfolio_annualized_volatility = diagnostics[
            "expected_annualized_volatility"
        ]
        portfolio_benchmark_covariance = float(
            benchmark_covariance @ eligible_weights
        )
        active_variance = float(
            portfolio_variance
            - 2 * portfolio_benchmark_covariance
            + benchmark_variance
        )
        residual_variance = float(
            portfolio_variance
            - portfolio_benchmark_covariance**2 / benchmark_variance
        )
        diagnostics.update(
            {
                "expected_benchmark_volatility": benchmark_annualized_volatility,
                "expected_volatility_ratio": (
                    portfolio_annualized_volatility
                    / benchmark_annualized_volatility
                ),
                "expected_tracking_error": math.sqrt(
                    max(0.0, active_variance) * args.annualization_days
                ),
                "portfolio_beta": portfolio_benchmark_covariance / benchmark_variance,
                "expected_idiosyncratic_volatility": math.sqrt(
                    max(0.0, residual_variance) * args.annualization_days
                ),
            }
        )
    return weights, eligible, diagnostics, expected_return_map


def performance_statistics(
    daily: pd.DataFrame,
    annualization_days: float,
    risk_free_rate: float,
    benchmark: str | None,
) -> dict[str, object]:
    returns = daily["portfolio_return"]
    observations = len(returns)
    ending_wealth = float(daily["portfolio_wealth"].iloc[-1])
    total_return = ending_wealth - 1.0
    annualized_return = ending_wealth ** (annualization_days / observations) - 1.0
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(annualization_days))
    daily_risk_free = math.expm1(math.log1p(risk_free_rate) / annualization_days)
    sharpe_ratio = (
        float(
            (returns.mean() - daily_risk_free)
            * annualization_days
            / annualized_volatility
        )
        if annualized_volatility > 0
        else float("nan")
    )
    wealth_with_start = pd.concat(
        [pd.Series([1.0]), daily["portfolio_wealth"].reset_index(drop=True)],
        ignore_index=True,
    )
    maximum_drawdown = float(
        (wealth_with_start / wealth_with_start.cummax() - 1.0).min()
    )
    summary: dict[str, object] = {
        "start_date": daily.index.min().date().isoformat(),
        "end_date": daily.index.max().date().isoformat(),
        "observations": observations,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "ending_wealth": ending_wealth,
        "rebalance_count": int(daily["is_rebalance"].sum()),
        "total_turnover": float(daily["turnover"].sum()),
        "average_turnover": float(
            daily.loc[daily["is_rebalance"], "turnover"].mean()
        ),
        "transaction_cost_fraction_sum": float(daily["transaction_cost"].sum()),
        "missing_held_return_count": int(daily["missing_held_returns"].sum()),
        "missing_benchmark_return_count": int(
            daily["benchmark_return_missing"].sum()
        ),
        "risk_free_rate": risk_free_rate,
        "benchmark": benchmark,
    }
    if benchmark is not None:
        benchmark_returns = daily["benchmark_return"]
        benchmark_wealth = float(daily["benchmark_wealth"].iloc[-1])
        active_returns = returns - benchmark_returns
        portfolio_benchmark_correlation = float(
            returns.corr(benchmark_returns)
        )
        benchmark_annualized_volatility = float(
            benchmark_returns.std(ddof=1) * math.sqrt(annualization_days)
        )
        benchmark_sharpe_ratio = (
            float(
                (benchmark_returns.mean() - daily_risk_free)
                * annualization_days
                / benchmark_annualized_volatility
            )
            if benchmark_annualized_volatility > 0
            else float("nan")
        )
        benchmark_wealth_with_start = pd.concat(
            [pd.Series([1.0]), daily["benchmark_wealth"].reset_index(drop=True)],
            ignore_index=True,
        )
        benchmark_maximum_drawdown = float(
            (
                benchmark_wealth_with_start
                / benchmark_wealth_with_start.cummax()
                - 1.0
            ).min()
        )
        realized_tracking_error = float(
            active_returns.std(ddof=1) * math.sqrt(annualization_days)
        )
        summary.update(
            {
                "benchmark_total_return": benchmark_wealth - 1.0,
                "benchmark_annualized_return": (
                    benchmark_wealth ** (annualization_days / observations) - 1.0
                ),
                "benchmark_annualized_volatility": benchmark_annualized_volatility,
                "benchmark_sharpe_ratio": benchmark_sharpe_ratio,
                "benchmark_maximum_drawdown": benchmark_maximum_drawdown,
                "portfolio_benchmark_correlation": portfolio_benchmark_correlation,
                "realized_tracking_error": realized_tracking_error,
                "relative_total_return": ending_wealth / benchmark_wealth - 1.0,
                "cumulative_return_difference": (
                    total_return - (benchmark_wealth - 1.0)
                ),
                "information_ratio": (
                    float(
                        active_returns.mean()
                        * annualization_days
                        / realized_tracking_error
                    )
                    if realized_tracking_error > 0
                    else float("nan")
                ),
                "benchmark_ending_wealth": benchmark_wealth,
            }
        )
    return summary


def compound_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def maximum_drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    wealth = pd.concat([pd.Series([1.0]), wealth.reset_index(drop=True)])
    return float((wealth / wealth.cummax() - 1.0).min())


def volatility_ratio_bucket(ratio: float) -> str:
    if ratio < 0.60:
        return "<0.60"
    if ratio < 0.80:
        return "0.60-<0.80"
    return ">=0.80"


def analyze_volatility_ratios(
    daily: pd.DataFrame,
    rebalances: pd.DataFrame,
    annualization_days: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure forward performance in non-overlapping ex-ante vol-ratio buckets."""
    period_rows: list[dict[str, object]] = []
    bucket_daily: dict[str, list[pd.DataFrame]] = {
        "<0.60": [],
        "0.60-<0.80": [],
        ">=0.80": [],
    }
    rebalance_dates = pd.to_datetime(rebalances["rebalance_date"])
    for position in range(len(rebalances)):
        rebalance = rebalances.iloc[position]
        start = rebalance_dates.iloc[position]
        complete = position + 1 < len(rebalances)
        end_exclusive = (
            rebalance_dates.iloc[position + 1]
            if complete
            else daily.index.max() + pd.Timedelta(days=1)
        )
        holding = daily.loc[(daily.index >= start) & (daily.index < end_exclusive)]
        if holding.empty:
            continue
        portfolio_returns = holding["portfolio_return"]
        benchmark_returns = holding["benchmark_return"]
        ratio = float(rebalance["expected_volatility_ratio"])
        bucket = volatility_ratio_bucket(ratio)
        portfolio_return = compound_return(portfolio_returns)
        benchmark_return = compound_return(benchmark_returns)
        period_rows.append(
            {
                "rebalance_date": start.date().isoformat(),
                "holding_end": holding.index.max().date().isoformat(),
                "complete_period": complete,
                "observations": len(holding),
                "volatility_ratio_bucket": bucket,
                "expected_volatility_ratio": ratio,
                "expected_portfolio_volatility": rebalance[
                    "expected_annualized_volatility"
                ],
                "expected_benchmark_volatility": rebalance[
                    "expected_benchmark_volatility"
                ],
                "expected_portfolio_beta": rebalance["portfolio_beta"],
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "active_return": portfolio_return - benchmark_return,
                "outperformed": portfolio_return > benchmark_return,
                "portfolio_realized_volatility": (
                    float(portfolio_returns.std(ddof=1) * math.sqrt(annualization_days))
                    if len(holding) > 1
                    else float("nan")
                ),
                "benchmark_realized_volatility": (
                    float(benchmark_returns.std(ddof=1) * math.sqrt(annualization_days))
                    if len(holding) > 1
                    else float("nan")
                ),
                "portfolio_maximum_drawdown": maximum_drawdown_from_returns(
                    portfolio_returns
                ),
                "benchmark_maximum_drawdown": maximum_drawdown_from_returns(
                    benchmark_returns
                ),
            }
        )
        if complete:
            bucket_daily[bucket].append(
                holding[["portfolio_return", "benchmark_return"]]
            )

    periods = pd.DataFrame(period_rows)
    completed = periods.loc[periods["complete_period"]].copy()
    if completed.empty:
        raise ValueError(
            "volatility-ratio analysis needs at least two rebalance dates"
        )

    daily_risk_free = math.expm1(math.log1p(risk_free_rate) / annualization_days)
    group_rows: list[dict[str, object]] = []
    for bucket in ["<0.60", "0.60-<0.80", ">=0.80"]:
        group_periods = completed.loc[
            completed["volatility_ratio_bucket"] == bucket
        ]
        if group_periods.empty:
            group_rows.append(
                {"volatility_ratio_bucket": bucket, "periods": 0}
            )
            continue
        pooled = pd.concat(bucket_daily[bucket]).sort_index()
        portfolio_returns = pooled["portfolio_return"]
        benchmark_returns = pooled["benchmark_return"]
        observations = len(pooled)
        portfolio_volatility = float(
            portfolio_returns.std(ddof=1) * math.sqrt(annualization_days)
        )
        benchmark_volatility = float(
            benchmark_returns.std(ddof=1) * math.sqrt(annualization_days)
        )
        active = group_periods["active_return"]
        active_standard_error = (
            float(active.std(ddof=1) / math.sqrt(len(active)))
            if len(active) > 1
            else float("nan")
        )
        up = pooled.loc[benchmark_returns > 0]
        down = pooled.loc[benchmark_returns < 0]
        group_rows.append(
            {
                "volatility_ratio_bucket": bucket,
                "periods": len(group_periods),
                "daily_observations": observations,
                "mean_expected_volatility_ratio": group_periods[
                    "expected_volatility_ratio"
                ].mean(),
                "average_portfolio_period_return": group_periods[
                    "portfolio_return"
                ].mean(),
                "average_benchmark_period_return": group_periods[
                    "benchmark_return"
                ].mean(),
                "average_active_period_return": active.mean(),
                "active_return_t_statistic": (
                    float(active.mean() / active_standard_error)
                    if active_standard_error > 0
                    else float("nan")
                ),
                "outperformance_rate": group_periods["outperformed"].mean(),
                "portfolio_annualized_return": (
                    (1.0 + portfolio_returns).prod()
                    ** (annualization_days / observations)
                    - 1.0
                ),
                "benchmark_annualized_return": (
                    (1.0 + benchmark_returns).prod()
                    ** (annualization_days / observations)
                    - 1.0
                ),
                "portfolio_annualized_volatility": portfolio_volatility,
                "benchmark_annualized_volatility": benchmark_volatility,
                "portfolio_sharpe_ratio": (
                    float(
                        (portfolio_returns.mean() - daily_risk_free)
                        * annualization_days
                        / portfolio_volatility
                    )
                    if portfolio_volatility > 0
                    else float("nan")
                ),
                "maximum_drawdown": maximum_drawdown_from_returns(
                    portfolio_returns
                ),
                "up_capture": (
                    float(up["portfolio_return"].mean() / up["benchmark_return"].mean())
                    if not up.empty and up["benchmark_return"].mean() != 0
                    else float("nan")
                ),
                "down_capture": (
                    float(
                        down["portfolio_return"].mean()
                        / down["benchmark_return"].mean()
                    )
                    if not down.empty and down["benchmark_return"].mean() != 0
                    else float("nan")
                ),
                "portfolio_benchmark_correlation": float(
                    portfolio_returns.corr(benchmark_returns)
                ),
            }
        )
    return periods, pd.DataFrame(group_rows)


def load_plotting(show_plot: bool):
    try:
        import matplotlib

        if not show_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib; install it with "
            "'python -m pip install matplotlib'"
        ) from exc
    return plt


def save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight")


def plot_performance(plt, daily: pd.DataFrame, path: Path, benchmark: str | None):
    panel_count = 3 if benchmark is not None else 2
    figure, axes = plt.subplots(
        panel_count, 1, figsize=(11, 3 * panel_count), sharex=True
    )
    portfolio_wealth = daily["portfolio_wealth"]
    axes[0].plot(daily.index, portfolio_wealth, label="Minimum variance")
    if benchmark is not None:
        benchmark_wealth = daily["benchmark_wealth"]
        axes[0].plot(daily.index, benchmark_wealth, label=benchmark)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wealth (log scale)")
    axes[0].set_title("Walk-forward performance")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    drawdown_axis = axes[-1]
    portfolio_peak = portfolio_wealth.cummax().clip(lower=1.0)
    drawdown_axis.plot(
        daily.index,
        100 * (portfolio_wealth / portfolio_peak - 1.0),
        label="Minimum variance",
    )
    if benchmark is not None:
        benchmark_peak = benchmark_wealth.cummax().clip(lower=1.0)
        axes[1].plot(
            daily.index,
            portfolio_wealth / benchmark_wealth,
            color="tab:purple",
        )
        axes[1].axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
        axes[1].set_ylabel("Relative wealth")
        axes[1].grid(alpha=0.25)
        drawdown_axis.plot(
            daily.index,
            100 * (benchmark_wealth / benchmark_peak - 1.0),
            label=benchmark,
        )
    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.set_xlabel("Date")
    drawdown_axis.grid(alpha=0.25)
    drawdown_axis.legend()
    figure.tight_layout()
    save_figure(figure, path)
    return figure


def plot_volatility(plt, periods: pd.DataFrame, path: Path, benchmark: str):
    completed = periods.loc[periods["complete_period"]].copy()
    dates = pd.to_datetime(completed["rebalance_date"])
    realized_ratio = (
        completed["portfolio_realized_volatility"]
        / completed["benchmark_realized_volatility"]
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(
        dates,
        100 * completed["expected_portfolio_volatility"],
        label="Portfolio forecast",
    )
    axes[0].plot(
        dates,
        100 * completed["expected_benchmark_volatility"],
        label=f"{benchmark} forecast",
    )
    axes[0].plot(
        dates,
        100 * completed["portfolio_realized_volatility"],
        linestyle="--",
        alpha=0.75,
        label="Portfolio subsequent realized",
    )
    axes[0].plot(
        dates,
        100 * completed["benchmark_realized_volatility"],
        linestyle="--",
        alpha=0.75,
        label=f"{benchmark} subsequent realized",
    )
    axes[0].set_ylabel("Annualized volatility (%)")
    axes[0].set_title("Forecast and subsequent realized volatility")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2)

    axes[1].plot(
        dates,
        completed["expected_volatility_ratio"],
        label="Forecast ratio",
    )
    axes[1].plot(
        dates,
        realized_ratio,
        linestyle="--",
        label="Subsequent realized ratio",
    )
    axes[1].axhline(0.60, color="grey", linewidth=0.8, linestyle=":")
    axes[1].axhline(0.80, color="grey", linewidth=0.8, linestyle=":")
    axes[1].set_ylabel("Portfolio / benchmark")
    axes[1].set_xlabel("Rebalance date")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    save_figure(figure, path)
    return figure


def plot_volatility_ratio_test(plt, periods: pd.DataFrame, path: Path):
    completed = periods.loc[periods["complete_period"]].copy()
    x = completed["expected_volatility_ratio"].to_numpy(dtype=float)
    y = 100 * completed["active_return"].to_numpy(dtype=float)
    colors = 100 * completed["benchmark_return"].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    points = axes[0].scatter(
        x, y, c=colors, cmap="coolwarm", edgecolor="black", linewidth=0.3
    )
    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        axes[0].plot(line_x, intercept + slope * line_x, color="black")
    axes[0].axhline(0.0, color="grey", linewidth=0.8)
    axes[0].axvline(0.60, color="grey", linewidth=0.8, linestyle=":")
    axes[0].axvline(0.80, color="grey", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("Ex-ante volatility ratio")
    axes[0].set_ylabel("Subsequent active return (%)")
    axes[0].set_title("Volatility ratio and forward active return")
    axes[0].grid(alpha=0.2)
    colorbar = figure.colorbar(points, ax=axes[0])
    colorbar.set_label("Subsequent benchmark return (%)")

    labels = ["<0.60", "0.60-<0.80", ">=0.80"]
    box_data = [
        100
        * completed.loc[
            completed["volatility_ratio_bucket"] == label, "active_return"
        ].to_numpy(dtype=float)
        for label in labels
    ]
    populated = [(label, values) for label, values in zip(labels, box_data) if len(values)]
    axes[1].boxplot(
        [values for _, values in populated],
        showmeans=True,
    )
    axes[1].set_xticks(
        range(1, len(populated) + 1),
        [label for label, _ in populated],
    )
    axes[1].axhline(0.0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("Ex-ante volatility-ratio bucket")
    axes[1].set_ylabel("Subsequent active return (%)")
    axes[1].set_title("Forward active-return distributions")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(figure, path)
    return figure


def plot_weights(
    plt,
    weights: pd.DataFrame,
    path: Path,
    top_weights: int,
):
    targets = weights.pivot(
        index="rebalance_date", columns="symbol", values="target_weight"
    ).fillna(0.0)
    targets.index = pd.to_datetime(targets.index)
    ranked = targets.abs().mean().sort_values(ascending=False)
    selected = ranked.index[:top_weights].tolist()
    plotted = targets[selected].copy()
    remaining = [symbol for symbol in targets.columns if symbol not in selected]
    if remaining:
        plotted["Other"] = targets[remaining].sum(axis=1)

    figure, axis = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, len(plotted.columns))
    )
    if (plotted < -1e-12).any().any():
        for symbol, color in zip(plotted.columns, colors):
            axis.plot(
                plotted.index,
                100 * plotted[symbol],
                label=symbol,
                color=color,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("Target weights at each rebalance")
    else:
        axis.stackplot(
            plotted.index,
            *[100 * plotted[column] for column in plotted.columns],
            labels=plotted.columns,
            colors=colors,
            alpha=0.85,
        )
        axis.set_title("Target-weight composition at each rebalance")
    axis.set_ylabel("Weight (%)")
    axis.set_xlabel("Rebalance date")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    save_figure(figure, path)
    return figure


def generate_plots(
    args: argparse.Namespace,
    stem: str,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    periods: pd.DataFrame | None,
) -> list[Path]:
    if not args.plot:
        return []
    plt = load_plotting(args.show_plot)
    figures = []
    paths: list[Path] = []
    for plot_name in args.plot:
        path = args.output_dir / (
            f"{stem}_backtest_{plot_name.replace('-', '_')}.{args.plot_format}"
        )
        if plot_name == "performance":
            figure = plot_performance(plt, daily, path, args.benchmark)
        elif plot_name == "volatility":
            figure = plot_volatility(plt, periods, path, args.benchmark)
        elif plot_name == "vol-ratio":
            figure = plot_volatility_ratio_test(plt, periods, path)
        else:
            figure = plot_weights(
                plt, weights, path, args.plot_top_weights
            )
        figures.append(figure)
        paths.append(path)
    if args.show_plot:
        plt.show()
    for figure in figures:
        plt.close(figure)
    return paths


def main(strategy: str = "minimum-variance") -> int:
    if strategy not in {"minimum-variance", "mean-variance"}:
        raise ValueError(f"unknown backtest strategy: {strategy}")
    overall_start = time.perf_counter()
    args = parse_args(strategy)
    if args.min_weight < 0 and args.max_gross is None:
        print("Warning: negative weights are enabled without a gross-exposure limit.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    result_stem = (
        f"{stem}_mean_variance"
        if strategy == "mean-variance" and args.name is None
        else stem
    )
    prices_path = (
        args.prices_file
        if args.prices_file is not None
        else args.output_dir / f"{stem}_adjusted_close.csv"
    )
    data_symbols = list(args.symbols)
    if args.benchmark is not None and args.benchmark not in data_symbols:
        data_symbols.append(args.benchmark)

    data_start = time.perf_counter()
    estimation_lookback = max(
        args.lookback,
        args.return_lookback or args.lookback,
    )
    prices, downloaded, unavailable, download_elapsed = ensure_saved_history(
        prices_path,
        data_symbols,
        estimation_lookback + args.backtest_days,
        args.code_dir,
        args.no_download_missing,
        critical_symbols=[args.benchmark] if args.benchmark is not None else None,
    )
    prices = prices.reindex(columns=data_symbols)
    returns = daily_returns(prices).dropna(how="all")
    maximum_backtest_days = len(returns) - estimation_lookback
    if maximum_backtest_days < 2:
        raise ValueError(
            f"only {len(returns)} return rows are available; need more than "
            f"the {estimation_lookback}-day estimation lookback"
        )
    out_of_sample_days = min(args.backtest_days, maximum_backtest_days)
    start_position = len(returns) - out_of_sample_days
    starting_eligible_symbols = None
    if args.min_symbols is not None:
        qualifying_position = None
        for position in range(start_position, len(returns)):
            training = returns.iloc[position - estimation_lookback:position]
            covariance_training, _, eligible = eligible_training_data(
                training, args
            )
            required = max(2, math.ceil(args.lookback * args.coverage))
            benchmark_qualified = (
                args.benchmark is None
                or covariance_training[args.benchmark].notna().sum() >= required
            )
            common_columns = list(eligible)
            if (
                args.benchmark is not None
                and args.benchmark not in common_columns
            ):
                common_columns.append(args.benchmark)
            has_common_sample = (
                benchmark_qualified
                and len(eligible) >= args.min_symbols
                and len(covariance_training[common_columns].dropna()) >= 2
            )
            if has_common_sample:
                qualifying_position = position
                starting_eligible_symbols = len(eligible)
                break
        if qualifying_position is None:
            raise ValueError(
                f"no backtest date has {args.min_symbols} symbols meeting the "
                "lookback coverage requirement"
            )
        start_position = qualifying_position
        out_of_sample_days = len(returns) - start_position
    data_elapsed = time.perf_counter() - data_start

    calculation_start = time.perf_counter()
    current_weights = np.zeros(len(args.symbols))
    target_weights = current_weights.copy()
    portfolio_wealth = 1.0
    benchmark_wealth = 1.0
    transaction_cost_rate = args.transaction_cost_bps / 10_000.0
    daily_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []

    for position in range(start_position, len(returns)):
        is_rebalance = (position - start_position) % args.rebalance_days == 0
        turnover = 0.0
        cost_fraction = 0.0
        if is_rebalance:
            training = returns.iloc[position - estimation_lookback:position]
            try:
                new_weights, eligible, diagnostics, expected_return_map = (
                    solve_rebalance(training, args)
                )
            except (ValueError, RuntimeError) as exc:
                date = returns.index[position].date().isoformat()
                raise ValueError(f"rebalance {date}: {exc}") from exc
            pretrade_weights = current_weights.copy()
            turnover = float(np.abs(new_weights - pretrade_weights).sum())
            cost_fraction = transaction_cost_rate * turnover
            if cost_fraction >= 1:
                raise ValueError(
                    "transaction cost is at least 100% of portfolio value on "
                    f"{returns.index[position].date()}"
                )
            target_weights = new_weights
            current_weights = new_weights.copy()
            eligible_set = set(eligible)
            for symbol, pretrade, target in zip(
                args.symbols, pretrade_weights, target_weights
            ):
                weight_row = {
                        "rebalance_date": returns.index[position].date().isoformat(),
                        "estimation_end": training.index[-1].date().isoformat(),
                        "symbol": symbol,
                        "eligible": symbol in eligible_set,
                        "pretrade_weight": pretrade,
                        "target_weight": target,
                        "trade_weight": target - pretrade,
                    }
                if args.strategy == "mean-variance":
                    weight_row["expected_annualized_return"] = (
                        expected_return_map.get(symbol, float("nan"))
                    )
                weight_rows.append(weight_row)
            rebalance_rows.append(
                {
                    "rebalance_date": returns.index[position].date().isoformat(),
                    "estimation_end": training.index[-1].date().isoformat(),
                    "turnover": turnover,
                    "transaction_cost": cost_fraction,
                    **diagnostics,
                }
            )

        day = returns.iloc[position]
        held = np.abs(current_weights) > 1e-12
        candidate_returns = day.reindex(args.symbols).to_numpy(dtype=float)
        missing = [
            symbol for symbol, value, is_held in zip(
                args.symbols, candidate_returns, held
            )
            if is_held and np.isnan(value)
        ]
        if missing and args.missing_return == "error":
            raise ValueError(
                f"missing held-asset returns on {returns.index[position].date()}: "
                + ", ".join(missing)
            )
        candidate_returns = np.nan_to_num(candidate_returns, nan=0.0)
        gross_portfolio_return = float(current_weights @ candidate_returns)
        if gross_portfolio_return <= -1:
            raise ValueError(
                f"portfolio lost at least 100% on {returns.index[position].date()}"
            )
        net_portfolio_return = (
            (1.0 - cost_fraction) * (1.0 + gross_portfolio_return) - 1.0
        )
        portfolio_wealth *= 1.0 + net_portfolio_return
        current_weights = (
            current_weights * (1.0 + candidate_returns)
            / (1.0 + gross_portfolio_return)
        )

        benchmark_return = np.nan
        benchmark_return_missing = False
        if args.benchmark is not None:
            benchmark_return = float(day[args.benchmark])
            if not np.isfinite(benchmark_return):
                benchmark_return_missing = True
                if args.missing_return == "error":
                    raise ValueError(
                        f"missing benchmark return on {returns.index[position].date()}"
                    )
                benchmark_return = 0.0
            benchmark_wealth *= 1.0 + benchmark_return
        daily_rows.append(
            {
                "date": returns.index[position],
                "portfolio_return": net_portfolio_return,
                "gross_portfolio_return": gross_portfolio_return,
                "portfolio_wealth": portfolio_wealth,
                "benchmark_return": benchmark_return,
                "benchmark_wealth": benchmark_wealth if args.benchmark else np.nan,
                "is_rebalance": is_rebalance,
                "turnover": turnover,
                "transaction_cost": cost_fraction,
                "missing_held_returns": len(missing),
                "benchmark_return_missing": benchmark_return_missing,
            }
        )

    daily = pd.DataFrame(daily_rows).set_index("date")
    weights = pd.DataFrame(weight_rows)
    rebalances = pd.DataFrame(rebalance_rows)
    summary = performance_statistics(
        daily, args.annualization_days, args.risk_free_rate, args.benchmark
    )
    summary.update(
        {
            "lookback": args.lookback,
            "strategy": args.strategy,
            "return_lookback": args.return_lookback,
            "return_estimator": args.return_estimator,
            "mean_shrinkage": args.mean_shrinkage,
            "ewma_halflife": args.ewma_halflife,
            "risk_aversion": args.risk_aversion,
            "target_return": args.target_return,
            "rebalance_days": args.rebalance_days,
            "requested_backtest_days": args.backtest_days,
            "minimum_starting_symbols": args.min_symbols,
            "starting_eligible_symbols": starting_eligible_symbols,
            "covariance_method": args.covariance,
            "coverage_required": args.coverage,
            "annualization_days": args.annualization_days,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "max_gross": args.max_gross,
            "max_tracking_error": args.max_tracking_error,
            "max_idiosyncratic_volatility": args.max_idio_vol,
            "volatility_ratio_analysis": args.vol_ratio_analysis,
            "transaction_cost_bps": args.transaction_cost_bps,
            "missing_return_policy": args.missing_return,
        }
    )
    volatility_ratio_periods = None
    volatility_ratio_groups = None
    needs_volatility_ratio_data = (
        args.vol_ratio_analysis
        or bool({"volatility", "vol-ratio"}.intersection(args.plot))
    )
    if needs_volatility_ratio_data:
        volatility_ratio_periods, volatility_ratio_groups = (
            analyze_volatility_ratios(
                daily,
                rebalances,
                args.annualization_days,
                args.risk_free_rate,
            )
        )
    calculation_elapsed = time.perf_counter() - calculation_start

    daily_path = args.output_dir / f"{result_stem}_backtest_daily.csv"
    weights_path = args.output_dir / f"{result_stem}_backtest_weights.csv"
    rebalances_path = args.output_dir / f"{result_stem}_backtest_rebalances.csv"
    summary_path = args.output_dir / f"{result_stem}_backtest_summary.csv"
    volatility_ratio_periods_path = (
        args.output_dir / f"{result_stem}_backtest_volatility_ratio_periods.csv"
    )
    volatility_ratio_groups_path = (
        args.output_dir / f"{result_stem}_backtest_volatility_ratio_summary.csv"
    )
    daily.to_csv(daily_path, index_label="date")
    weights.to_csv(weights_path, index=False)
    rebalances.to_csv(rebalances_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    if args.vol_ratio_analysis:
        volatility_ratio_periods.to_csv(
            volatility_ratio_periods_path, index=False
        )
        volatility_ratio_groups.to_csv(
            volatility_ratio_groups_path, index=False
        )
    plotting_start = time.perf_counter()
    plot_paths = generate_plots(
        args,
        result_stem,
        daily,
        weights,
        volatility_ratio_periods,
    )
    plotting_elapsed = time.perf_counter() - plotting_start

    print(f"Using prices: {prices_path}")
    if downloaded:
        print("Downloaded symbols: " + ", ".join(downloaded))
    if unavailable:
        print(
            "Unavailable symbols excluded when coverage permits: "
            + ", ".join(unavailable)
        )
    if args.min_symbols is not None:
        print(
            f"Backtest starts {returns.index[start_position].date()} when "
            f"{starting_eligible_symbols} of {len(args.symbols)} symbols meet "
            "the coverage requirement."
        )
    if out_of_sample_days < args.backtest_days:
        print(
            f"Using {out_of_sample_days} available backtest days instead of "
            f"the requested {args.backtest_days}."
        )
    if args.strategy == "mean-variance":
        formulation = (
            f"risk aversion {args.risk_aversion:g}"
            if args.risk_aversion is not None
            else f"target return {args.target_return:.2%}"
        )
        print(
            f"\nWalk-forward mean-variance backtest: "
            f"{args.lookback}-day covariance lookback, "
            f"{args.return_lookback}-day {args.return_estimator} return estimate, "
            f"{formulation}; rebalance every {args.rebalance_days} trading days"
        )
    else:
        print(
            f"\nWalk-forward backtest: {args.lookback}-day lookback, "
            f"rebalance every {args.rebalance_days} trading days"
        )
    print(
        f"{summary['start_date']} to {summary['end_date']}; "
        f"{summary['observations']} observations; "
        f"{summary['rebalance_count']} rebalances"
    )
    print(
        f"total return: {100 * summary['total_return']:.2f}%; "
        f"CAGR: {100 * summary['annualized_return']:.2f}%; "
        f"annualized volatility: {100 * summary['annualized_volatility']:.2f}%; "
        f"Sharpe ratio: {summary['sharpe_ratio']:.3f}; "
        f"maximum drawdown: {100 * summary['maximum_drawdown']:.2f}%"
    )
    if args.benchmark is not None:
        print(
            f"benchmark {args.benchmark}: total return: "
            f"{100 * summary['benchmark_total_return']:.2f}%; "
            f"CAGR: {100 * summary['benchmark_annualized_return']:.2f}%; "
            f"volatility: {100 * summary['benchmark_annualized_volatility']:.2f}%; "
            f"Sharpe ratio: {summary['benchmark_sharpe_ratio']:.3f}; "
            f"maximum drawdown: "
            f"{100 * summary['benchmark_maximum_drawdown']:.2f}%"
        )
        print(
            f"relative total return: "
            f"{100 * summary['relative_total_return']:.2f}%; "
            f"cumulative return difference: "
            f"{100 * summary['cumulative_return_difference']:+.2f} pp; "
            f"realized tracking error: {100 * summary['realized_tracking_error']:.2f}%; "
            f"information ratio: {summary['information_ratio']:.3f}; "
            f"correlation: {summary['portfolio_benchmark_correlation']:.3f}"
        )
    print(
        f"total turnover: {100 * summary['total_turnover']:.2f}%; "
        f"average per rebalance: {100 * summary['average_turnover']:.2f}%; "
        f"transaction-cost sum: "
        f"{100 * summary['transaction_cost_fraction_sum']:.2f}%"
    )
    if (
        summary["missing_held_return_count"]
        or summary["missing_benchmark_return_count"]
    ):
        print(
            f"missing returns treated as zero: held assets "
            f"{summary['missing_held_return_count']}; benchmark "
            f"{summary['missing_benchmark_return_count']}"
        )
    if args.vol_ratio_analysis:
        print(
            "\nforward performance by ex-ante portfolio/benchmark "
            "volatility ratio:"
        )
        print("(completed, non-overlapping rebalance periods; net portfolio returns)")
        ratio_display = volatility_ratio_groups.copy()
        ratio_display = ratio_display.rename(
            columns={
                "volatility_ratio_bucket": "ratio",
                "mean_expected_volatility_ratio": "mean_ratio",
                "average_portfolio_period_return": "avg_port_ret",
                "average_benchmark_period_return": "avg_bench_ret",
                "average_active_period_return": "avg_active_ret",
                "active_return_t_statistic": "active_t",
                "outperformance_rate": "win_rate",
                "portfolio_annualized_volatility": "portfolio_vol",
                "benchmark_annualized_volatility": "benchmark_vol",
                "portfolio_sharpe_ratio": "sharpe",
                "maximum_drawdown": "max_drawdown",
            }
        )
        ratio_columns = [
            "ratio", "periods", "mean_ratio", "avg_port_ret",
            "avg_bench_ret", "avg_active_ret", "active_t", "win_rate",
            "portfolio_vol", "benchmark_vol", "sharpe", "max_drawdown",
            "up_capture", "down_capture",
        ]
        ratio_display = ratio_display.reindex(columns=ratio_columns)
        for column in [
            "avg_port_ret", "avg_bench_ret", "avg_active_ret", "win_rate",
            "portfolio_vol", "benchmark_vol", "max_drawdown", "up_capture",
            "down_capture",
        ]:
            ratio_display[column] = ratio_display[column].map(
                lambda value: f"{100 * value:.2f}%" if pd.notna(value) else "n/a"
            )
        for column in ["mean_ratio", "active_t", "sharpe"]:
            ratio_display[column] = ratio_display[column].map(
                lambda value: f"{value:.3f}" if pd.notna(value) else "n/a"
            )
        print(ratio_display.to_string(index=False))
    print("\nrebalance diagnostics:")
    display = rebalances.copy()
    for column in ["turnover", "expected_annualized_volatility", "gross_exposure"]:
        display[column] = (100 * display[column]).map(lambda value: f"{value:.2f}%")
    optional_percent_columns = [
        "expected_portfolio_return",
        "expected_benchmark_volatility",
        "expected_tracking_error",
        "expected_idiosyncratic_volatility",
    ]
    for column in optional_percent_columns:
        if column in display:
            display[column] = (100 * display[column]).map(lambda value: f"{value:.2f}%")
    for column in ["expected_volatility_ratio", "portfolio_beta"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    print(display.to_string(index=False))
    for path in [daily_path, weights_path, rebalances_path, summary_path]:
        print(f"Wrote {path}")
    if args.vol_ratio_analysis:
        print(f"Wrote {volatility_ratio_periods_path}")
        print(f"Wrote {volatility_ratio_groups_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    print(f"Data elapsed: {data_elapsed:.3f} seconds")
    if downloaded:
        print(f"Download elapsed: {download_elapsed:.3f} seconds")
    print(f"Calculations elapsed: {calculation_elapsed:.3f} seconds")
    if plot_paths:
        print(f"Plotting elapsed: {plotting_elapsed:.3f} seconds")
    print(f"Overall elapsed: {time.perf_counter() - overall_start:.3f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
