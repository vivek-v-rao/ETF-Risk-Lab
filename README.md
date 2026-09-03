# ETF constituent betas

`etf_constituent_betas.py` reads the equity rows from an iShares-style holdings
CSV, downloads adjusted daily closes from Yahoo Finance, and computes stock
betas against one or more ETFs.

```powershell
python .\etf_constituent_betas.py .\TOPT_holdings.csv --etfs VOO XLE --days 252 --multiple
```

Several holdings files can share one merged price download while retaining
fund-specific calculation outputs:

```powershell
python .\etf_constituent_betas.py .\IVV_holdings.csv .\TOPT_holdings.csv `
  --etfs VOO --days 63 126 252
```

For a multi-file run, merged price files use the joined holdings stems (for
example `IVV_holdings__TOPT_holdings_adjusted_close.csv`). Beta, volatility, and
idiosyncratic-volatility result files continue to use each individual holdings
file's stem.

`--days` accepts one or more daily **return** windows. For example,
`--days 63 126 252` downloads enough data for the 252-day window and calculates
all three windows from that one dataset. Up to one additional price observation
is needed for each window. Univariate results are sorted in ascending beta order
within each window and ETF. The
reported grid uses `[-1, 0, 1, 2]` as cut points, including underflow and overflow
bins: `< -1`, `[-1, 0)`, `[0, 1)`, `[1, 2)`, and `>= 2`.

Outputs go to `output/` by default:

- `<holdings>_adjusted_close.csv`: all adjusted-close data downloaded
- `<holdings>_constituent_adjusted_close.csv`: constituent adjusted closes only
- `<holdings>_etf_adjusted_close.csv`: factor ETF adjusted closes only
- `<holdings>_univariate_betas.csv`: sorted univariate betas and observation counts
- `<holdings>_beta_grid_counts.csv`: counts by ETF and beta interval
- `<holdings>_multiple_betas.csv`: optional joint-regression results

Use `--no-save-data` to suppress the downloaded-price CSV, `--output-dir` to
choose another directory, and `--end-date YYYY-MM-DD` for a historical run. The
end date is exclusive; by default the script uses today and therefore excludes
the possibly incomplete current trading session.

To rerun the calculations without downloading data, use the default combined
price file:

```powershell
python .\etf_constituent_betas.py .\TOPT_holdings.csv --etfs VOO XLE --days 252 --multiple --use-saved-data
```

An explicit saved CSV can also be supplied:

```powershell
python .\etf_constituent_betas.py .\TOPT_holdings.csv --etfs VOO XLE --use-saved-data .\archive\prices.csv
```

Historical and idiosyncratic volatility calculations reuse the same price data:

```powershell
python .\etf_constituent_betas.py .\TOPT_holdings.csv --etfs VOO XLE `
  --days 63 126 252 --vol --idiovol both --use-saved-data
```

`--vol` produces annualized historical constituent volatility. `--idiovol` by
itself calculates both modes; it also accepts an explicit `univariate`, `multiple`,
or `both`. The multiple calculation uses all factor ETFs and does not require
`--multiple`. With only one factor ETF, the redundant multiple result is omitted
and the calculation is reported as univariate. Console summaries are annualized percentages,
while CSV values are annualized decimals. Change the default square-root-of-252
annualization with `--annualization-days`.

Additional optional outputs are:

- `<holdings>_historical_volatilities.csv`
- `<holdings>_univariate_idiosyncratic_volatilities.csv`
- `<holdings>_multiple_idiosyncratic_volatilities.csv`

Use `--sector` to add sector classifications to constituent result files and to
print beta and requested volatility statistics separately for every sector. It
also writes `<holdings>_sector_statistics.csv` in long format; values in that CSV
retain their source units (betas or annualized decimal volatility).

## Minimum-variance portfolios

`minimum_variance_portfolio.py` reads the saved adjusted-close data and constructs
long-only minimum-variance portfolios independently for every holdings file and
window. It uses Ledoit-Wolf covariance shrinkage by default:

```powershell
python .\minimum_variance_portfolio.py .\IVV_holdings.csv .\TOPT_holdings.csv `
  --prices-file .\output\IVV_holdings__TOPT_holdings_adjusted_close.csv `
  --days 126 252 --max-weight 0.05
```

Omit `--prices-file` when the price file uses the normal single- or multi-holdings
prefix. Use `--covariance sample` for the ordinary sample covariance estimator.
`--coverage` defaults to 0.90; securities below that fraction of available window
returns are excluded, after which covariance is estimated on common dates. SciPy
is required for optimization and scikit-learn is required for Ledoit-Wolf:

```powershell
python -m pip install scipy scikit-learn
```

Matplotlib is optional and is required only for plot output:

```powershell
python -m pip install matplotlib
```

Portfolio outputs are:

- `<holdings>_minimum_variance_weights.csv`
- `<holdings>_minimum_variance_summary.csv`

For a portfolio specified directly as Yahoo symbols, with no holdings file, use
`minimum_variance_symbols.py`:

```powershell
python .\minimum_variance_symbols.py IVV XLE XLP --days 63 126 252 `
  --max-weight 0.60
```

The script downloads enough adjusted-close history for the longest window, saves
it as `output/IVV__XLE__XLP_adjusted_close.csv`, and constructs a separate
long-only minimum-variance portfolio for every requested window. Subsequent runs
reuse that cache; `--no-download-missing` enforces offline use. Use
`--prices-file` to select another cache, `--covariance sample` for sample
covariance, and `--coverage` to control the minimum usable history per symbol.
Results are written to:

- `<symbols>_minimum_variance_weights.csv`
- `<symbols>_minimum_variance_summary.csv`

Symbols can instead be read from a CSV containing a `symbol` or `ticker` column,
or from a text file containing one symbol per line:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\symbols.csv --days 126 252
```

`--symbols-file` may be repeated and combined with positional symbols. Symbols
from all sources are normalized and deduplicated in their original order. For a
quick test on a large universe, `--max-symbols N` (or `--max-sym N`) keeps only
the first N unique symbols after that processing:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\symbols.csv --max-symbols 10
```

With one symbols file and no positional symbols, its filename stem becomes the
default output prefix. For example, `sector_spdr_symbols.txt` produces
`sector_spdr_symbols_adjusted_close.csv` and corresponding short result names.
Use `--name` to select an explicit prefix, especially for mixed or multiple
symbol sources:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\symbols.csv `
  --name sector_etfs --days 126 252
```

When `--prices-file` is provided without positional symbols or `--symbols-file`,
the program infers the universe from the price column headers. This makes a saved
cache directly reusable, with `--max-symbols` applied to the inferred column order.
The price filename without `_adjusted_close` becomes the result prefix unless
`--name` is supplied:

```powershell
python .\minimum_variance_symbols.py --days 126 252 --max-sym 4 `
  --prices-file .\output\SPY__XLB__XLC__XLE_adjusted_close.csv
```

The symbol table also reports each asset's annualized covariance-model
volatility, correlation to the optimized portfolio, cumulative window return,
and Sharpe ratio. The portfolio summary reports its cumulative return and Sharpe
ratio as well. Sharpe ratios use arithmetic mean daily excess returns and the
selected covariance model's annualized volatility. The annual effective
risk-free rate defaults to 4%; set it as a decimal with, for example,
`--risk-free-rate 0.03`. Portfolio returns assume daily rebalancing. All reported
returns, correlations, and Sharpe ratios are in-sample statistics rather than an
out-of-sample backtest.

Use `--corrmat` to print an estimator-consistent correlation matrix for every
window. Each matrix contains all eligible assets plus a final `*MIN_VOL*` row and
column, and is also saved as `<name>_<days>d_correlation_matrix.csv`. The console
asset table always ends with the same marked portfolio row for comparison; this
synthetic row is not added to the weights CSV:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\sector_spdr_symbols.txt `
  --days 126 252 --corrmat
```

The direct-symbol optimizer is long-only by default. Set a negative per-symbol
lower bound with `--min-weight` (also accepted as `--min-wgt` or `--min_wgt`) to
permit shorts. Because short positions can create substantial leverage, an
optional gross-exposure cap is available:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\sector_spdr_symbols.txt `
  --days 126 --min-weight -1.0 --max-weight 1.0 --max-gross 2.0
```

Weights still sum to 100%; `--max-gross 2.0` limits absolute weights to 200% in
total. Console and summary CSV output report long, short, and gross exposure plus
the largest long and short positions. If negative weights are enabled without
`--max-gross`, the program prints a leverage warning.

To minimize absolute portfolio volatility while limiting annualized tracking
error relative to an ETF, provide a benchmark and a decimal tracking-error cap:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\sector_spdr_symbols.txt `
  --days 126 252 --benchmark SPY --max-tracking-error 0.03
```

The benchmark is added to the saved price cache automatically when necessary;
`--no-download-missing` retains its strict offline meaning. The benchmark need
not be an investable portfolio symbol. Output reports benchmark volatility and
return, portfolio tracking error, active period return, information ratio, and
whether the constraint binds. Tracking error uses the selected covariance
estimator and `--annualization-days`. If the cap cannot be achieved under the
weight and gross-exposure constraints, the error reports the minimum achievable
tracking error. `--benchmark` without a cap reports the same comparison metrics
without constraining the minimum-volatility portfolio.

An idiosyncratic-volatility constraint lets benchmark beta vary instead of
fixing it at one as tracking error does. It limits the annualized residual
volatility after an optimal single-benchmark beta hedge:

```powershell
python .\minimum_variance_symbols.py --symbols-file .\sector_spdr_symbols.txt `
  --days 126 --benchmark SPY --min-weight -1 --max-gross 2 `
  --max-idio-vol 0.03
```

The investable weights are optimized with their requested leverage first; the
benchmark hedge is diagnostic and is not included in the weight-sum or
`--max-gross` constraint. Output reports portfolio beta, idiosyncratic
volatility, the implied benchmark hedge, and gross exposure after adding that
hedge. If the benchmark is also investable, the hedge is netted against its
existing portfolio weight when calculating hedged gross exposure. Hedged net
exposure is reported as well. The summary also reports whether the
idiosyncratic-volatility cap binds. If it is infeasible—including when combined
with a tracking-error cap—the error
reports the minimum achievable residual volatility under the other constraints.

## Walk-forward minimum-variance backtests

`minimum_variance_backtest.py` applies the same covariance estimators, weight
bounds, gross limit, and benchmark-risk constraints without look-ahead. This
example estimates weights from the preceding 126 returns and rebalances every 63
trading days over a requested five-year evaluation period. `--backtest-days` is
an upper bound; if less history exists, the program uses every available return
after reserving the initial lookback window:

```powershell
python .\minimum_variance_backtest.py `
  --symbols-file .\sector_spdr_symbols.txt `
  --lookback 126 --rebalance-days 63 --backtest-days 1260 `
  --benchmark SPY --transaction-cost-bps 5
```

Use `--min-symbols N` (or `--min-sym N`) to move the starting date forward until
at least N investable symbols meet `--coverage` over a complete lookback window:

```powershell
python .\minimum_variance_backtest.py `
  --symbols-file .\sector_spdr_symbols.txt `
  --lookback 63 --rebalance-days 63 --backtest-days 10000 `
  --min-sym 10 --benchmark SPY
```

This is a starting-universe condition only. Later temporary gaps continue to use
the ordinary coverage and missing-return policies.

With a benchmark, `--vol-ratio-analysis` tests whether the strategy subsequently
does better when its forecast volatility is especially low relative to the
benchmark:

```powershell
python .\minimum_variance_backtest.py `
  --symbols-file .\sector_spdr_symbols.txt `
  --lookback 63 --rebalance-days 63 --backtest-days 10000 `
  --min-sym 10 --benchmark SPY --vol-ratio-analysis
```

Each completed, non-overlapping holding period is assigned using the volatility
ratio forecast at its rebalance date, avoiding look-ahead bias. The fixed buckets
are `<0.60`, `0.60-<0.80`, and `>=0.80`. Console and CSV output include
average portfolio, benchmark, and active period returns; win rate; realized
volatilities; Sharpe ratio; maximum drawdown; daily up/down capture; and an
unadjusted t-statistic for mean active period return. The program additionally
writes `<name>_backtest_volatility_ratio_periods.csv` and
`<name>_backtest_volatility_ratio_summary.csv`. The last, unfinished holding
period appears in the period file but is excluded from grouped statistics.

Use repeatable `--plot` options to save backtest figures:

```powershell
python .\minimum_variance_backtest.py `
  --symbols-file .\sector_spdr_symbols.txt --benchmark SPY `
  --plot performance --plot volatility --plot vol-ratio --plot weights
```

The available plot names are `performance`, `volatility`, `vol-ratio`, and
`weights`; `--plot all` selects all four. Figures are saved without opening a
window by default. Add `--show-plot` to display all requested figures after they
have been saved. `--show-plot` without `--plot` is an error. Use
`--plot-format png|pdf|svg` to select the file type and `--plot-top-weights N`
to control how many individual positions appear in the weights chart; remaining
positions are combined as `Other`.

Weights estimated through a given close first earn the following trading day's
return. Between rebalances, positions drift with asset returns instead of being
implicitly rebalanced every day. One-way turnover is the sum of absolute trades;
the first portfolio purchase is included. Transaction costs are deducted on each
rebalance date.

The backtest writes:

- `<name>_backtest_daily.csv`: daily net/gross returns, wealth, benchmark, costs,
  and rebalance indicators
- `<name>_backtest_weights.csv`: pre-trade, target, and trade weights by rebalance
- `<name>_backtest_rebalances.csv`: covariance and constraint diagnostics
- `<name>_backtest_summary.csv`: portfolio and benchmark return, volatility,
  Sharpe ratio, maximum drawdown, compounded relative return, cumulative-return
  difference in decimal units, realized correlation, turnover, and cost statistics

`--benchmark`, `--max-tracking-error`, `--max-idio-vol`, `--min-weight`,
`--max-weight`, and `--max-gross` have the same meanings as in the single-period
program and are re-evaluated using only each rebalance's training window. Price
history is expanded automatically when the existing cache is too short; use
`--no-download-missing` for strict offline operation.

Yahoo data occasionally contains isolated missing observations. The default
`--missing-return zero` uses a stale-price assumption and reports the number of
affected held-asset and benchmark returns. Use `--missing-return error` for a
strict run.

Results based on a current constituent or symbol list have survivorship bias and
are not a historical constituent backtest. Adjusted-close returns also omit
taxes, bid/ask effects beyond the specified transaction cost, short borrow fees,
and financing costs. These outputs are research estimates, not executable
performance records.

## Walk-forward mean-variance backtests

`mean_variance_backtest.py` uses the same walk-forward accounting, constraints,
benchmark analysis, plots, price cache, and output structure, but estimates
expected returns at every rebalance and solves either

`minimize (risk_aversion / 2) * variance - expected_return`

or minimum variance subject to an expected-return target. For example:

```powershell
python .\mean_variance_backtest.py `
  --symbols-file .\sector_spdr_symbols.txt `
  --lookback 126 --return-lookback 252 `
  --return-estimator shrinkage --mean-shrinkage 0.50 `
  --risk-aversion 3 --max-weight 0.25 --benchmark SPY
```

`--return-estimator` accepts `historical`, `ewma`, or `shrinkage`. Shrinkage,
the default, pulls each arithmetic historical mean toward the cross-sectional
mean; `--mean-shrinkage 0` is the raw historical estimator and `1` gives every
asset the cross-sectional mean. For EWMA, configure `--ewma-halflife`.
`--return-lookback` defaults to the covariance `--lookback`.

If neither formulation is supplied, `--risk-aversion 3` is used. Alternatively,
replace it with an annualized decimal target such as `--target-return 0.10`.
The two options are mutually exclusive, and an infeasible target reports the
maximum expected return allowed by the weight and gross-exposure constraints.

Mean-variance result names include `_mean_variance` by default, while saved price
files retain the ordinary symbol-list name and are shared with the minimum-
variance backtest. The weights CSV includes each eligible asset's estimated
annualized return. Expected-return estimates are especially noisy, so compare
results across estimators and lookbacks and include realistic transaction costs.
The shared implementation is in `portfolio_backtest.py`; the two strategy files
are small command-line entry points.

Sector-sleeve optimization reduces the covariance dimension while keeping stock
weights proportional to the source ETF weights within each sector:

```powershell
python .\minimum_variance_portfolio.py .\IVV_holdings.csv `
  --prices-file .\output\IVV_holdings__TOPT_holdings_adjusted_close.csv `
  --days 126 252 --portfolio-level sectors --max-sector-weight 0.25
```

Use `--portfolio-level both` to run stock- and sector-level optimizations. Stocks
that fail `--coverage` or lack a positive source holdings weight are excluded;
their weight is redistributed proportionally among eligible stocks in the same
sector. Console and CSV output compare optimized weights with normalized ETF
holdings weights; `active_weight` is minimum-variance weight minus ETF weight.
Both ETF and active weights are also included for derived constituents. Sector
mode writes:

- `<holdings>_sector_minimum_variance_weights.csv`
- `<holdings>_sector_minimum_variance_constituent_weights.csv`
- `<holdings>_sector_minimum_variance_summary.csv`

### Fixed legacy position and new-money completion

Use `--legacy-symbol` to keep an existing ETF position fixed and optimize only
the new-money sleeve. For example, this keeps 90% in IVV and allocates the
remaining 10% across sector sleeves:

```powershell
python .\minimum_variance_portfolio.py .\IVV_holdings.csv --days 126 `
  --portfolio-level sectors --max-sector-weight 0.30 `
  --legacy-symbol IVV --legacy-weight 0.90
```

The primary saved-price file must include the legacy symbol. If its prices are
stored in a separate CSV, provide `--legacy-prices-file` as well.
`--legacy-symbol` alone uses a 0.90 legacy weight; specifying `--legacy-weight`
without `--legacy-symbol` is an error.

If a required symbol column is absent, the portfolio script downloads only the
missing symbol through the shared Yahoo helper, aligns it to the saved file's last
date, and updates that CSV without removing existing columns. Existing unavailable
constituents remain subject to coverage exclusion. Use `--no-download-missing` for
strict offline operation.

Completion output distinguishes sleeve weights (summing to 100% of new money),
portfolio weights (summing to the new-money fraction), and total look-through
weights after adding the legacy ETF exposure. Summaries report legacy volatility,
standalone sleeve volatility, completed-portfolio volatility, and estimated
volatility reduction versus investing everything in the legacy ETF.

For several fixed positions, use a CSV with signed final-portfolio weights:

```csv
ticker,weight
IVV,0.90
XLE,-0.10
```

```powershell
python .\minimum_variance_portfolio.py .\IVV_holdings.csv --days 126 `
  --portfolio-level sectors --legacy-positions .\legacy_positions.csv
```

Negative fixed weights represent shorts or hedges. Their net sum determines the
new-money fraction, while gross, long, and short exposures are reported
separately. The net fixed weight must be below 1; net-zero signed portfolios are
supported. The optimized new-money sleeve remains long-only. For multiple legacy
assets, direct weights are reported but sector look-through is omitted unless
holdings mappings become available for every legacy asset.

The shared `etf_data.py` module owns holdings and symbol-list parsing,
Yahoo-symbol normalization, multi-fund merging, price downloading/loading,
history-aware cache completion, validation, and return calculation.
`portfolio_optimization.py` owns coherent candidate/benchmark covariance
estimation and the bounded optimizer used by the single-period, holdings, and
backtest programs. The command-line scripts keep their simulation and reporting
logic separate.
