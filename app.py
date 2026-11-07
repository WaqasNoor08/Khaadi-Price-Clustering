"""Streamlit dashboard for the Khaadi price clustering project.

Run:  streamlit run app.py
Reads the CSVs produced by cluster_prices.py (output/) and the candidate cache (data/).
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Khaadi Price Tiers", page_icon="👗", layout="wide")

BASE = Path(__file__).parent
TIER_ORDER = ["Budget", "Mid-range", "Premium"]


def find(name: str) -> Path:
    """Look for a file in output/, data/ or next to this script."""
    for folder in ("output", "data", "."):
        p = BASE / folder / name
        if p.exists():
            return p
    st.error(f"File not found: {name}. Run cluster_prices.py first or put the CSV in output/.")
    st.stop()


@st.cache_data
def load():
    items = pd.read_csv(find("item_clusters.csv"))
    summary = pd.read_csv(find("cluster_summary.csv"))
    budget = pd.read_csv(find("budget_plan.csv"))
    matrix = pd.read_csv(find("brand_tier_matrix.csv"))
    cand = pd.read_csv(find("video_candidates.csv"), encoding="utf-8-sig")
    return items, summary, budget, matrix, cand


items, summary, budget, matrix, cand = load()
color = alt.Scale(domain=TIER_ORDER, range=["#4C78A8", "#F58518", "#54A24B"])

st.title("Khaadi: price tiers, sentiment and marketing budget")
st.caption("YouTube comments, VADER sentiment, KMeans price tiers. Preliminary: small sample.")

total_comments = int(summary["comments"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidate videos", len(cand))
c2.metric("Products analysed", len(items))
c3.metric("Comments", total_comments)
c4.metric("Overall positive", f"{items['pos'].sum() / total_comments * 100:.1f}%")

st.warning(
    "Tier sentiment is statistically indistinguishable (chi-square = 0.82, p = 0.935). "
    "With 40-92 comments per tier this is not evidence that price has no effect. "
    "The budget split is a demonstration of the method, not a recommendation."
)

tab1, tab2, tab3, tab4 = st.tabs(["Tiers & sentiment", "Budget", "Products", "Data"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Tier summary")
        show = summary[
            ["price_tier", "price_min", "price_max", "items", "comments",
             "positive_pct", "neutral_pct", "negative_pct", "positive_rate_lower95"]
        ].rename(columns={"positive_rate_lower95": "positive_lower95"})
        st.dataframe(show, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Sentiment by tier")
        long = summary.melt(
            id_vars="price_tier",
            value_vars=["positive_pct", "neutral_pct", "negative_pct"],
            var_name="sentiment", value_name="pct",
        )
        long["sentiment"] = long["sentiment"].str.replace("_pct", "")
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("price_tier:N", sort=TIER_ORDER, title=None),
                y=alt.Y("pct:Q", title="% of comments"),
                color=alt.Color("sentiment:N", scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#54A24B", "#BAB0AC", "#E45756"])),
                tooltip=["price_tier", "sentiment", alt.Tooltip("pct:Q", format=".1f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Price vs. positive share (each dot = one product)")
    scatter = (
        alt.Chart(items)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("price:Q", scale=alt.Scale(type="log"), title="Price (PKR, log scale)"),
            y=alt.Y("pos_pct:Q", title="% positive comments"),
            size=alt.Size("comments:Q", title="Comments"),
            color=alt.Color("price_tier:N", scale=color, title="Tier"),
            tooltip=["item", "price", "comments", "pos_pct", "price_tier"],
        )
    )
    st.altair_chart(scatter, use_container_width=True)

with tab2:
    st.subheader("Budget allocation")
    total = st.number_input("Total marketing budget (PKR)", min_value=0,
                            value=1_000_000, step=50_000)
    plan = budget.copy()
    plan["budget_pkr"] = (plan["budget_share_pct"] / plan["budget_share_pct"].sum() * total).round(0)
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(plan, hide_index=True, use_container_width=True)
        st.caption("Shares come from budget_plan.csv (Wilson lower bound x comment volume). "
                   "Changing the budget only rescales the shares.")
    with right:
        bar = (
            alt.Chart(plan)
            .mark_bar()
            .encode(
                x=alt.X("price_tier:N", sort=TIER_ORDER, title=None),
                y=alt.Y("budget_pkr:Q", title="PKR"),
                color=alt.Color("price_tier:N", scale=color, legend=None),
                tooltip=["price_tier", alt.Tooltip("budget_pkr:Q", format=",.0f"),
                         "budget_share_pct"],
            )
        )
        st.altair_chart(bar, use_container_width=True)

with tab3:
    st.subheader("Products")
    tiers = st.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
    min_c = st.slider("Minimum comments", 1, int(items["comments"].max()), 1)
    view = items[items["price_tier"].isin(tiers) & (items["comments"] >= min_c)]
    st.dataframe(
        view[["item", "price", "price_tier", "comments", "pos", "neg", "pos_pct"]]
        .sort_values("comments", ascending=False),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"{len(view)} products shown. Note: the 'Khaadi IPO' video is not a product.")

with tab4:
    st.subheader("Candidate videos")
    only_non_promo = st.checkbox("Hide promotional videos", value=True)
    data = cand[~cand["promo"]] if only_non_promo else cand
    st.write(f"{len(data)} of {len(cand)} videos")
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.subheader("Brand x tier matrix")
    st.dataframe(matrix, hide_index=True, use_container_width=True)
