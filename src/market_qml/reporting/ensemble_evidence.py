"""Explicit ensemble comparisons against single-model and equal-weight controls."""

import pandas as pd


def compare_ensemble_performance(ranking_metrics, risk_metrics):
    split = ranking_metrics.loc[ranking_metrics["scope"].eq("split")].copy()
    overall = ranking_metrics.loc[ranking_metrics["scope"].eq("overall")].set_index("model_name")
    ensembles = [name for name in overall.index if "ensemble" in name]
    singles = [name for name in overall.index if "ensemble" not in name]
    columns = ["model_name", "rank_ic", "best_single_model", "best_single_rank_ic", "rank_ic_delta_vs_single", "rank_ic_delta_vs_equal", "fold_win_share_vs_single", "cumulative_net_excess_return", "net_max_drawdown", "improved_stably_after_costs"]
    if not ensembles or not singles:
        return pd.DataFrame(columns=columns)
    best_single = overall.loc[singles, "rank_information_coefficient"].astype(float).idxmax()
    best_ic = float(overall.loc[best_single, "rank_information_coefficient"])
    single_splits = split.loc[split["model_name"].eq(best_single)].set_index("split_id")["rank_information_coefficient"]
    risks = risk_metrics.loc[risk_metrics["scope"].eq("overall")].set_index("model_name")
    rows = []
    for name in ensembles:
        ic = float(overall.loc[name, "rank_information_coefficient"])
        equal_name = name.replace("constrained_stack", "simple_average")
        equal_ic = float(overall.loc[equal_name, "rank_information_coefficient"]) if equal_name in overall.index else pd.NA
        ensemble_splits = split.loc[split["model_name"].eq(name)].set_index("split_id")["rank_information_coefficient"]
        paired = pd.concat([ensemble_splits.rename("ensemble"), single_splits.rename("single")], axis=1).dropna()
        win_share = float((paired["ensemble"] > paired["single"]).mean()) if len(paired) else pd.NA
        net_excess = risks.loc[name, "cumulative_net_excess_return"] if name in risks.index else pd.NA
        drawdown = risks.loc[name, "net_max_drawdown"] if name in risks.index else pd.NA
        rows.append({"model_name": name, "rank_ic": ic, "best_single_model": best_single, "best_single_rank_ic": best_ic, "rank_ic_delta_vs_single": ic - best_ic, "rank_ic_delta_vs_equal": ic - equal_ic if pd.notna(equal_ic) else pd.NA, "fold_win_share_vs_single": win_share, "cumulative_net_excess_return": net_excess, "net_max_drawdown": drawdown, "improved_stably_after_costs": bool(ic > best_ic and pd.notna(win_share) and win_share >= 0.6 and pd.notna(net_excess) and net_excess > 0)})
    return pd.DataFrame(rows, columns=columns)
