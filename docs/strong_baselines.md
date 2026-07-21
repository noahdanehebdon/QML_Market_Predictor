# Strong tabular and learning-to-rank baselines

The strong classical comparison includes an XGBoost classifier and a date-grouped
LambdaMART ranker. XGBoost is maintained and Apache-2.0 licensed. Classification uses
binary log loss, inverse-date-frequency sample weights, and chronological inner-fold
early stopping. Ranking uses `rank:ndcg`; each prediction date is one query group and
continuous returns are converted to within-date relevance levels using training rows.

Neither model observes the outer validation fold during selection. Runtime, best
iteration, calibration, rank IC, spread, turnover, and fold stability flow through the
existing walk-forward diagnostics. Claims of improvement require repeated-fold results
and comparison against the repository's linear, momentum, and random controls; a single
point estimate is not treated as evidence of superiority.

References:

- https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/learning_to_rank.html
- https://github.com/dmlc/xgboost/blob/master/LICENSE
