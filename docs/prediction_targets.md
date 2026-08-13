# Prediction target research protocol

Candidate labels use information available at the close of trading date `t` and
the close at `t + H`, where `H` is 5, 10, 20, or 60 trading days. The benchmark
is SPY. A row is missing when either future stock or same-date benchmark return
is unavailable. Neutral-zone labels are additionally missing when absolute
excess return is at or below 0.5%.

The workflow compares benchmark-relative binary, neutral-zone binary,
continuous excess return, trailing-volatility-normalized excess return,
same-date cross-sectional rank, and—when point-in-time sector membership is
provided—sector-relative return and sector rank. Volatility uses only returns
known through `t`. Cross-sectional transforms use only labels sharing the same
outcome start date; they do not mix future dates.

Run:

```powershell
python -m scripts.research_prediction_targets
```

Every candidate uses a purge equal to its horizon. The script partitions off
the latest 252 trading dates plus a five-day embargo before computing any
diagnostic. It reports class balance, label turnover, autocorrelation,
period-to-period stability, missingness, economic magnitude, and rank IC between
the horizon-lagged target and the current outcome. That lag equals the target's
purge, so the IC diagnostic never uses an overlapping outcome window.
Selection is deterministic and development-only. Chronological periods before
the final development period form three inner folds; the final development
period is the outer validation fold. Earlier periods are calibration history,
and each target keeps its horizon-specific purge. The manifest records zero
locked-test rows inspected. Sector-relative candidates require a `sector`
column (or another point-in-time column passed with `--sector-column`). Static
present-day sector mappings must not be backfilled into historical rows.

The five-day SPY sign label remains the comparison baseline. A chosen target is
promoted only when its score exceeds the baseline by the configured practical
margin, it has at least two chronological validation periods, no more than 20%
missing labels, and positive rank IC in at least half of those periods.
Classification candidates must also retain at least 60% balance quality. A
failure of any gate records a null result rather than promoting a fragile
winner. The locked period is opened only after configurations are frozen and
access is logged through `scripts.log_locked_test_access`.

`residualized_forward_excess_return_{horizon}d` removes available same-date beta,
volatility, liquidity, sector, and size exposures for stock-specific ranking. Label
tables also retain `return_integrity_valid`, `return_integrity_status`, and a robust
cross-sectional return z-score. Invalid outcomes remain auditable but are excluded
from modeling datasets rather than clipped after model selection.
