"""Shared covariance estimation and bounded portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def covariance_matrix(sample: pd.DataFrame, method: str) -> np.ndarray:
    if method == "ledoit-wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for --covariance ledoit-wolf"
            ) from exc
        covariance = LedoitWolf().fit(sample.to_numpy()).covariance_
    elif method == "sample":
        covariance = np.cov(sample.to_numpy(), rowvar=False, ddof=1)
    else:
        raise ValueError(f"unknown covariance method: {method}")
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    return (covariance + covariance.T) / 2


def candidate_benchmark_covariances(
    sample: pd.DataFrame,
    candidate_columns: list[str],
    method: str,
    benchmark: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Estimate a coherent candidate/benchmark covariance system."""
    ordered = list(candidate_columns)
    if benchmark is not None and benchmark not in ordered:
        ordered.append(benchmark)
    joint_covariance = covariance_matrix(sample[ordered], method)
    candidate_count = len(candidate_columns)
    candidate_covariance = joint_covariance[:candidate_count, :candidate_count]
    if benchmark is None:
        return candidate_covariance, None, None
    benchmark_index = ordered.index(benchmark)
    benchmark_covariance = joint_covariance[:candidate_count, benchmark_index]
    benchmark_variance = float(joint_covariance[benchmark_index, benchmark_index])
    return candidate_covariance, benchmark_covariance, benchmark_variance


def completion_covariances(
    sample: pd.DataFrame,
    candidate_columns: list[str],
    method: str,
    legacy_weights: dict[str, float] | None,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    """Estimate candidate covariance and fixed legacy contribution covariance."""
    if not legacy_weights:
        return covariance_matrix(sample[candidate_columns], method), None, None
    legacy_columns = list(legacy_weights)
    ordered = [*legacy_columns, *candidate_columns]
    joint = covariance_matrix(sample[ordered], method)
    legacy_count = len(legacy_columns)
    fixed_weights = np.array([legacy_weights[symbol] for symbol in legacy_columns])
    legacy_covariance = joint[:legacy_count, :legacy_count]
    legacy_candidate_covariance = joint[:legacy_count, legacy_count:]
    candidate_covariance = joint[legacy_count:, legacy_count:]
    fixed_variance = float(fixed_weights @ legacy_covariance @ fixed_weights)
    fixed_cross_covariance = fixed_weights @ legacy_candidate_covariance
    return candidate_covariance, fixed_variance, fixed_cross_covariance


def optimize_bounded(
    covariance: np.ndarray,
    min_weight: float,
    max_weight: float,
    annualization_days: float,
    fixed_cross_covariance: np.ndarray | None = None,
    new_money_weight: float = 1.0,
    max_gross: float | None = None,
    benchmark_covariance: np.ndarray | None = None,
    benchmark_variance: float | None = None,
    max_tracking_error: float | None = None,
    max_idiosyncratic_volatility: float | None = None,
    expected_returns: np.ndarray | None = None,
    risk_aversion: float | None = None,
    target_return: float | None = None,
) -> tuple[np.ndarray, object | None]:
    """Optimize variance or mean-variance utility under portfolio constraints."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("scipy is required for portfolio optimization") from exc

    asset_count = covariance.shape[0]
    if expected_returns is not None:
        expected_returns = np.asarray(expected_returns, dtype=float)
        if expected_returns.shape != (asset_count,):
            raise ValueError("expected returns have the wrong dimension")
        if not np.isfinite(expected_returns).all():
            raise ValueError("expected returns must be finite")
    if risk_aversion is not None and target_return is not None:
        raise ValueError("risk aversion and target return are alternative formulations")
    if risk_aversion is not None and (
        not np.isfinite(risk_aversion) or risk_aversion <= 0
    ):
        raise ValueError("risk aversion must be finite and positive")
    if target_return is not None and not np.isfinite(target_return):
        raise ValueError("target return must be finite")
    if (risk_aversion is not None or target_return is not None) and expected_returns is None:
        raise ValueError("expected returns are required for mean-variance optimization")
    if min_weight > max_weight:
        raise ValueError("minimum weight cannot exceed maximum weight")
    if min_weight * asset_count > 1 + 1e-12 or max_weight * asset_count < 1 - 1e-12:
        raise ValueError(
            f"weight bounds [{min_weight:g}, {max_weight:g}] are infeasible for "
            f"{asset_count} eligible assets; they must contain "
            f"{1 / asset_count:.6f}"
        )
    if max_gross is not None and max_gross < 1 - 1e-12:
        raise ValueError("--max-gross must be at least 1 because weights sum to 1")
    if max_tracking_error is not None and max_tracking_error < 0:
        raise ValueError("maximum tracking error cannot be negative")
    if (
        max_idiosyncratic_volatility is not None
        and max_idiosyncratic_volatility < 0
    ):
        raise ValueError("maximum idiosyncratic volatility cannot be negative")
    if (
        max_tracking_error is not None
        or max_idiosyncratic_volatility is not None
    ):
        if benchmark_covariance is None or benchmark_variance is None:
            raise ValueError(
                "benchmark covariance and variance are required for benchmark-risk limits"
            )
        benchmark_covariance = np.asarray(benchmark_covariance, dtype=float)
        if benchmark_covariance.shape != (asset_count,):
            raise ValueError("benchmark covariance has the wrong dimension")
    if max_idiosyncratic_volatility is not None and benchmark_variance <= 0:
        raise ValueError(
            "benchmark variance must be positive for an idiosyncratic-volatility limit"
        )
    if new_money_weight <= 0:
        raise ValueError("new-money weight must be positive")
    if asset_count == 1:
        if target_return is not None and expected_returns[0] < target_return - 1e-12:
            raise ValueError(
                f"target return {target_return:.2%} is infeasible; maximum "
                f"expected return is {expected_returns[0]:.2%}"
            )
        if max_tracking_error is not None:
            active_variance = float(
                covariance[0, 0]
                - 2 * benchmark_covariance[0]
                + benchmark_variance
            )
            tracking_error = np.sqrt(
                max(0.0, active_variance) * annualization_days
            )
            if tracking_error > max_tracking_error + 1e-8:
                raise ValueError(
                    f"maximum tracking error {max_tracking_error:.2%} is infeasible; "
                    f"minimum achievable tracking error is {tracking_error:.2%}"
                )
        if max_idiosyncratic_volatility is not None:
            residual_variance = float(
                covariance[0, 0]
                - benchmark_covariance[0] ** 2 / benchmark_variance
            )
            idiosyncratic_volatility = np.sqrt(
                max(0.0, residual_variance) * annualization_days
            )
            if (
                idiosyncratic_volatility
                > max_idiosyncratic_volatility + 1e-8
            ):
                raise ValueError(
                    "maximum idiosyncratic volatility "
                    f"{max_idiosyncratic_volatility:.2%} is infeasible; minimum "
                    "achievable idiosyncratic volatility is "
                    f"{idiosyncratic_volatility:.2%}"
                )
        return np.ones(1), None

    annual_covariance = covariance * annualization_days
    if fixed_cross_covariance is None:
        fixed_cross_covariance = np.zeros(asset_count)
    annual_cross_covariance = fixed_cross_covariance * annualization_days
    completion_ratio = 1 / new_money_weight
    initial = np.full(asset_count, 1 / asset_count)

    if risk_aversion is None:
        def weight_objective(weights: np.ndarray) -> float:
            return float(
                weights @ annual_covariance @ weights
                + 2 * completion_ratio * annual_cross_covariance @ weights
            )

        def weight_gradient(weights: np.ndarray) -> np.ndarray:
            return (
                2 * annual_covariance @ weights
                + 2 * completion_ratio * annual_cross_covariance
            )
    else:
        def weight_objective(weights: np.ndarray) -> float:
            variance_term = float(
                weights @ annual_covariance @ weights
                + 2 * completion_ratio * annual_cross_covariance @ weights
            )
            return risk_aversion * variance_term / 2 - float(
                expected_returns @ weights
            )

        def weight_gradient(weights: np.ndarray) -> np.ndarray:
            return risk_aversion * (
                annual_covariance @ weights
                + completion_ratio * annual_cross_covariance
            ) - expected_returns

    if max_gross is None:
        initial_values = initial
        bounds = [(min_weight, max_weight)] * asset_count
        base_constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda values: values.sum() - 1.0,
                "jac": lambda values: np.ones_like(values),
            }
        ]
    else:
        # Auxiliary absolute-weight variables make the gross constraint linear.
        initial_values = np.concatenate([initial, np.abs(initial)])
        identity = np.eye(asset_count)
        zero_row = np.zeros(asset_count)
        base_constraints = [
            {
                "type": "eq",
                "fun": lambda values: values[:asset_count].sum() - 1.0,
                "jac": lambda values: np.concatenate(
                    [np.ones(asset_count), zero_row]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda values: values[asset_count:] - values[:asset_count],
                "jac": lambda values: np.concatenate([-identity, identity], axis=1),
            },
            {
                "type": "ineq",
                "fun": lambda values: values[asset_count:] + values[:asset_count],
                "jac": lambda values: np.concatenate([identity, identity], axis=1),
            },
            {
                "type": "ineq",
                "fun": lambda values: max_gross - values[asset_count:].sum(),
                "jac": lambda values: np.concatenate([zero_row, -np.ones(asset_count)]),
            },
        ]
        bounds = [
            *[(min_weight, max_weight)] * asset_count,
            *[(0.0, max_gross)] * asset_count,
        ]

    if target_return is not None:
        def negative_expected_return(values: np.ndarray) -> float:
            return -float(expected_returns @ values[:asset_count])

        def negative_expected_return_gradient(values: np.ndarray) -> np.ndarray:
            gradient = -expected_returns
            if max_gross is not None:
                gradient = np.concatenate([gradient, np.zeros(asset_count)])
            return gradient

        maximum_return_result = minimize(
            negative_expected_return,
            initial_values,
            jac=negative_expected_return_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=base_constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        if not maximum_return_result.success:
            raise RuntimeError(
                "maximum-return feasibility optimization failed: "
                f"{maximum_return_result.message}"
            )
        maximum_expected_return = -negative_expected_return(
            maximum_return_result.x
        )
        if maximum_expected_return < target_return - 1e-10:
            raise ValueError(
                f"target return {target_return:.2%} is infeasible; maximum "
                f"expected return is {maximum_expected_return:.2%}"
            )
        initial_values = maximum_return_result.x

        def target_return_constraint(values: np.ndarray) -> float:
            return float(expected_returns @ values[:asset_count] - target_return)

        def target_return_gradient(values: np.ndarray) -> np.ndarray:
            gradient = expected_returns
            if max_gross is not None:
                gradient = np.concatenate([gradient, np.zeros(asset_count)])
            return gradient

        base_constraints.append(
            {
                "type": "ineq",
                "fun": target_return_constraint,
                "jac": target_return_gradient,
            }
        )

    def value_weights(values: np.ndarray) -> np.ndarray:
        return values[:asset_count]

    def extend_gradient(weight_values: np.ndarray) -> np.ndarray:
        if max_gross is None:
            return weight_values
        return np.concatenate([weight_values, np.zeros(asset_count)])

    def objective(values: np.ndarray) -> float:
        return weight_objective(value_weights(values))

    def gradient(values: np.ndarray) -> np.ndarray:
        return extend_gradient(weight_gradient(value_weights(values)))

    constraints = list(base_constraints)
    if max_tracking_error is not None:
        annual_benchmark_variance = float(benchmark_variance) * annualization_days
        annual_benchmark_covariance = benchmark_covariance * annualization_days

        def active_variance(values: np.ndarray) -> float:
            weights = value_weights(values)
            return float(
                weights @ annual_covariance @ weights
                - 2 * annual_benchmark_covariance @ weights
                + annual_benchmark_variance
            )

        def active_variance_gradient(values: np.ndarray) -> np.ndarray:
            weights = value_weights(values)
            return extend_gradient(
                2 * annual_covariance @ weights - 2 * annual_benchmark_covariance
            )

        minimum_tracking_result = minimize(
            active_variance,
            initial_values,
            jac=active_variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=base_constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        if not minimum_tracking_result.success:
            raise RuntimeError(
                "minimum-tracking-error feasibility optimization failed: "
                f"{minimum_tracking_result.message}"
            )
        minimum_tracking_error = np.sqrt(
            max(0.0, active_variance(minimum_tracking_result.x))
        )
        if minimum_tracking_error > max_tracking_error + 1e-8:
            raise ValueError(
                f"maximum tracking error {max_tracking_error:.2%} is infeasible; "
                f"minimum achievable tracking error is {minimum_tracking_error:.2%}"
            )
        initial_values = minimum_tracking_result.x
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values: (
                    max_tracking_error**2 - active_variance(values)
                ),
                "jac": lambda values: -active_variance_gradient(values),
            }
        )

    if max_idiosyncratic_volatility is not None:
        residual_covariance = covariance - np.outer(
            benchmark_covariance, benchmark_covariance
        ) / float(benchmark_variance)
        residual_covariance = (residual_covariance + residual_covariance.T) / 2
        annual_residual_covariance = residual_covariance * annualization_days

        def residual_variance(values: np.ndarray) -> float:
            weights = value_weights(values)
            return float(weights @ annual_residual_covariance @ weights)

        def residual_variance_gradient(values: np.ndarray) -> np.ndarray:
            weights = value_weights(values)
            return extend_gradient(2 * annual_residual_covariance @ weights)

        minimum_idio_result = minimize(
            residual_variance,
            initial_values,
            jac=residual_variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        if not minimum_idio_result.success:
            raise RuntimeError(
                "minimum-idiosyncratic-volatility feasibility optimization failed: "
                f"{minimum_idio_result.message}"
            )
        minimum_idiosyncratic_volatility = np.sqrt(
            max(0.0, residual_variance(minimum_idio_result.x))
        )
        if (
            minimum_idiosyncratic_volatility
            > max_idiosyncratic_volatility + 1e-8
        ):
            raise ValueError(
                "maximum idiosyncratic volatility "
                f"{max_idiosyncratic_volatility:.2%} is infeasible; minimum "
                "achievable idiosyncratic volatility is "
                f"{minimum_idiosyncratic_volatility:.2%}"
            )
        initial_values = minimum_idio_result.x
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values: (
                    max_idiosyncratic_volatility**2
                    - residual_variance(values)
                ),
                "jac": lambda values: -residual_variance_gradient(values),
            }
        )

    result = minimize(
        objective,
        initial_values,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        strategy = "mean-variance" if expected_returns is not None else "minimum-variance"
        raise RuntimeError(f"{strategy} optimization failed: {result.message}")
    if max_tracking_error is not None:
        achieved_tracking_error = np.sqrt(
            max(0.0, active_variance(result.x))
        )
        if achieved_tracking_error > max_tracking_error + 1e-7:
            raise RuntimeError(
                "minimum-variance optimization violated the tracking-error limit: "
                f"{achieved_tracking_error:.2%} > {max_tracking_error:.2%}"
            )
    if max_idiosyncratic_volatility is not None:
        achieved_idiosyncratic_volatility = np.sqrt(
            max(0.0, residual_variance(result.x))
        )
        if (
            achieved_idiosyncratic_volatility
            > max_idiosyncratic_volatility + 1e-7
        ):
            raise RuntimeError(
                "minimum-variance optimization violated the idiosyncratic-volatility "
                f"limit: {achieved_idiosyncratic_volatility:.2%} > "
                f"{max_idiosyncratic_volatility:.2%}"
            )
    return result.x[:asset_count], result


def optimize_long_only(
    covariance: np.ndarray,
    max_weight: float,
    annualization_days: float,
    fixed_cross_covariance: np.ndarray | None = None,
    new_money_weight: float = 1.0,
) -> tuple[np.ndarray, object | None]:
    """Minimize variance with nonnegative weights and sum(weights) equal to one."""
    return optimize_bounded(
        covariance,
        0.0,
        max_weight,
        annualization_days,
        fixed_cross_covariance,
        new_money_weight,
    )
