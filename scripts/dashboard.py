"""Launch the optional Streamlit model-results dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from market_qml.reporting.dashboard import (
    latest_signal_report,
    model_comparison,
    portfolio_series,
    qml_experiment_summary,
    top_ranked_stocks,
)


REPORTS_DIR = Path("reports")


st.set_page_config(page_title="QML Market Predictor", page_icon="📈", layout="wide")
st.title("QML Market Predictor")
st.caption("Local research dashboard — results are not financial advice.")

signals = latest_signal_report(REPORTS_DIR)
comparison = model_comparison(REPORTS_DIR)
portfolio = portfolio_series(REPORTS_DIR)
top_stocks = top_ranked_stocks(REPORTS_DIR)
qml_summary = qml_experiment_summary(REPORTS_DIR)

st.header("Latest signal report")
if signals.empty:
    st.info(
        "No daily signal report is available. Run "
        "`python scripts/generate_daily_signal_report.py` to create it."
    )
else:
    signal_date = signals["date"].max() if "date" in signals else None
    if signal_date is not None:
        st.caption(f"Signal date: {signal_date.date().isoformat()}")
    st.dataframe(signals, width="stretch", hide_index=True)

st.header("Model comparison")
if comparison.empty:
    st.warning("No classical model comparison artifact was found.")
else:
    st.dataframe(comparison, width="stretch", hide_index=True)

st.header("Portfolio results")
if portfolio.empty:
    st.warning("No portfolio return artifact was found.")
else:
    selected_models = st.multiselect(
        "Models",
        sorted(portfolio["model_name"].dropna().unique()),
        default=sorted(portfolio["model_name"].dropna().unique()),
    )
    visible = portfolio.loc[portfolio["model_name"].isin(selected_models)]
    cumulative, drawdowns = st.columns(2)
    with cumulative:
        st.subheader("Cumulative net returns")
        st.line_chart(
            visible.pivot_table(
                index="date",
                columns="model_name",
                values="cumulative_net_return",
                aggfunc="last",
            )
        )
    with drawdowns:
        st.subheader("Drawdowns")
        st.line_chart(
            visible.pivot_table(
                index="date",
                columns="model_name",
                values="drawdown",
                aggfunc="last",
            )
        )

st.header("Top-ranked stocks")
if top_stocks.empty:
    st.warning("No signal or prediction artifact was found.")
else:
    st.dataframe(top_stocks, width="stretch", hide_index=True)

st.header("QML experiment summaries")
if qml_summary.empty:
    st.warning("No QML aggregate metrics artifact was found.")
else:
    st.dataframe(qml_summary, width="stretch", hide_index=True)

with st.expander("Experiment notes"):
    notes_path = Path("docs/qml_experiments.md")
    if notes_path.exists():
        st.markdown(notes_path.read_text(encoding="utf-8"))
    else:
        st.info("No QML experiment notes were found.")
