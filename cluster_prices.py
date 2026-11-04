"""Price clustering and marketing-budget allocation from product comments.

Input CSV columns: brand, item, price, comment
(``sentiment`` and ``score`` are optional; if missing they are computed with VADER.)

Pipeline
--------
1. Aggregate comments per item and cluster the items by price (KMeans on log-price).
2. Summarise sentiment per price tier and test whether tiers really differ (chi-square).
3. Split a marketing budget across tiers using a conservative positive-rate estimate.
4. Save tables and charts.

Usage
-----
    python cluster_prices.py --input data/comments.csv --budget 1000000
    python cluster_prices.py --input data/comments.csv --k 3 --min-share 0.05
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render charts without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SENTIMENT_COLORS = {"Positive": "#2e9e5b", "Neutral": "#9aa0a6", "Negative": "#d64545"}
MIN_ITEMS_REQUIRED = 4        # below this clustering is impossible
MIN_ITEMS_RECOMMENDED = 30    # below this the tier statistics are unreliable
MIN_TIERS, MAX_TIERS = 3, 6   # range searched when k is chosen automatically
TIER_NAMES = {
    3: ["Budget", "Mid-range", "Premium"],
    4: ["Budget", "Lower-mid", "Upper-mid", "Premium"],
}


# --------------------------------------------------------------------------- #
# Loading and sentiment
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Strip links and @mentions and collapse whitespace."""
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment columns unless the data already has them."""
    if {"sentiment", "score"}.issubset(df.columns):
        return df
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    df["score"] = [analyzer.polarity_scores(clean_text(t))["compound"] for t in df["comment"]]
    df["sentiment"] = np.select(
        [df["score"] >= 0.05, df["score"] <= -0.05], ["Positive", "Negative"], default="Neutral"
    )
    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"brand", "item", "price", "comment"} - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "comment"])
    df = df[df["price"] > 0].reset_index(drop=True)
    return add_sentiment(df)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def cluster_items(items: pd.DataFrame, k: int | None, seed: int = 42) -> tuple[pd.DataFrame, int]:
    """Cluster items by price. Prices are right-skewed, so log-price is used.

    When ``k`` is None the number of tiers (3..6) with the best silhouette score
    is chosen; at least three tiers are required because two are too coarse
    for marketing decisions.
    """
    x = np.log(items[["price"]].to_numpy())

    if k is None:
        upper = min(MAX_TIERS, len(items) - 1)
        scores = {}
        for candidate in range(MIN_TIERS, upper + 1):
            labels = KMeans(n_clusters=candidate, n_init=10, random_state=seed).fit_predict(x)
            scores[candidate] = silhouette_score(x, labels)
        k = max(scores, key=scores.get)
        print("Silhouette score by k:", {c: round(float(s), 3) for c, s in scores.items()},
              f"-> k={k}")

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    # Re-number clusters by price so that tier 0 is the cheapest.
    order = np.argsort(model.cluster_centers_.ravel())
    rank = {old: new for new, old in enumerate(order)}
    names = TIER_NAMES.get(k, [f"Tier {i + 1}" for i in range(k)])
    items = items.copy()
    items["cluster_id"] = [rank[c] for c in model.labels_]
    items["price_tier"] = [names[c] for c in items["cluster_id"]]
    return items, k


# --------------------------------------------------------------------------- #
# Statistics and budget
# --------------------------------------------------------------------------- #
def wilson_lower_bound(positives: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval for a proportion.

    Used instead of the raw positive rate so that tiers with few comments are
    not over-trusted.
    """
    if n == 0:
        return 0.0
    p = positives / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def summarise_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment statistics per price tier, ordered from cheapest to most expensive."""
    rows = []
    for _, group in df.groupby("cluster_id"):
        n = len(group)
        positives = int((group["sentiment"] == "Positive").sum())
        rows.append({
            "price_tier": group["price_tier"].iloc[0],
            "price_min": group["price"].min(),
            "price_max": group["price"].max(),
            "items": group["item"].nunique(),
            "comments": n,
            "positive_pct": round(positives / n * 100, 1),
            "neutral_pct": round((group["sentiment"] == "Neutral").mean() * 100, 1),
            "negative_pct": round((group["sentiment"] == "Negative").mean() * 100, 1),
            "avg_score": round(group["score"].mean(), 3),
            "positive_rate_lower95": round(wilson_lower_bound(positives, n) * 100, 1),
        })
    return pd.DataFrame(rows)


def allocate_budget(summary: pd.DataFrame, total: float, min_share: float) -> pd.DataFrame:
    """Split ``total`` across tiers.

    Weight = (conservative positive rate) x sqrt(comments). Every tier first
    receives ``min_share`` of the budget; the rest is shared by weight.
    """
    if min_share * len(summary) >= 1:
        raise SystemExit("--min-share is too large for the number of tiers.")
    weight = (summary["positive_rate_lower95"] / 100) * np.sqrt(summary["comments"])
    share = min_share + (1 - min_share * len(summary)) * (weight / weight.sum())
    summary = summary.copy()
    summary["budget_share_pct"] = (share * 100).round(1)
    summary["budget_pkr"] = (share * total).round(-3)
    return summary


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_price_clusters(items: pd.DataFrame, tiers: list[str], outdir: Path) -> None:
    palette = sns.color_palette("viridis", len(tiers))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for colour, tier in zip(palette, tiers):
        ax.hist(items.loc[items["price_tier"] == tier, "price"], bins=25,
                alpha=0.8, color=colour, label=tier)
    ax.set(xlabel="Price (PKR)", ylabel="Items", title="Price clusters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "price_clusters.png", dpi=150)
    plt.close(fig)


def plot_sentiment_by_tier(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    tiers = summary["price_tier"].tolist()
    mix = pd.crosstab(df["price_tier"], df["sentiment"], normalize="index").reindex(tiers) * 100
    mix = mix[[c for c in SENTIMENT_COLORS if c in mix.columns]]
    ax = mix.plot(kind="bar", stacked=True, figsize=(8, 4.5),
                  color=[SENTIMENT_COLORS[c] for c in mix.columns])
    ax.set_xticklabels(
        [f"{r.price_tier}\nRs {r.price_min:,.0f}-{r.price_max:,.0f}" for r in summary.itertuples()],
        rotation=0)
    ax.set(ylabel="% of comments", xlabel="", title="Sentiment by price tier")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(outdir / "sentiment_by_tier.png", dpi=150)
    plt.close()


def plot_brand_tier_heatmap(matrix: pd.DataFrame, tiers: list[str], min_comments: int,
                            outdir: Path) -> None:
    """Positive % per brand and tier; cells with too few comments are hidden."""
    visible = matrix[matrix["comments"] >= min_comments]
    if visible.empty:
        return
    table = visible.pivot(index="brand", columns="price_tier", values="positive_pct")
    table = table[[t for t in tiers if t in table.columns]]
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(table, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                cbar_kws={"label": "% positive"})
    plt.title(f"Positive % by brand and tier (cells with >= {min_comments} comments)")
    plt.tight_layout()
    plt.savefig(outdir / "brand_tier_heatmap.png", dpi=150)
    plt.close()


def plot_budget(summary: pd.DataFrame, total: float, outdir: Path) -> None:
    palette = sns.color_palette("viridis", len(summary))
    ax = summary.plot(x="price_tier", y="budget_pkr", kind="bar", legend=False,
                      color=palette, figsize=(8, 4.5))
    for i, row in enumerate(summary.itertuples()):
        ax.text(i, row.budget_pkr, f"{row.budget_share_pct}%", ha="center", va="bottom")
    ax.set(xlabel="", ylabel="PKR", title=f"Suggested marketing budget split (total Rs {total:,.0f})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "budget_allocation.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Price clustering and budget allocation")
    p.add_argument("--input", default="data/comments.csv")
    p.add_argument("--outdir", default="output")
    p.add_argument("--k", type=int, default=None, help="number of price tiers (default: automatic)")
    p.add_argument("--budget", type=float, default=1_000_000, help="total marketing budget (PKR)")
    p.add_argument("--min-share", type=float, default=0.05, help="minimum budget share per tier")
    p.add_argument("--min-comments", type=int, default=20,
                   help="minimum comments for a brand x tier heatmap cell")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)

    # Item-level table and clustering.
    items = (df.groupby(["brand", "item"])
               .agg(price=("price", "median"), comments=("comment", "size"),
                    pos=("sentiment", lambda s: int((s == "Positive").sum())),
                    neg=("sentiment", lambda s: int((s == "Negative").sum())))
               .reset_index())
    items["pos_pct"] = (items["pos"] / items["comments"] * 100).round(1)

    if len(items) < MIN_ITEMS_REQUIRED:
        raise SystemExit(f"Only {len(items)} items: at least {MIN_ITEMS_REQUIRED} are required.")
    if len(items) < MIN_ITEMS_RECOMMENDED:
        print(f"WARNING: only {len(items)} items (>= {MIN_ITEMS_RECOMMENDED} recommended). "
              "Treat the results as a preliminary estimate.")

    items, k = cluster_items(items, args.k)
    items.to_csv(outdir / "item_clusters.csv", index=False)
    df = df.merge(items[["brand", "item", "cluster_id", "price_tier"]], on=["brand", "item"])

    # Tier statistics, significance test and budget.
    summary = summarise_tiers(df)
    chi2, p_value, _, _ = chi2_contingency(pd.crosstab(df["price_tier"], df["sentiment"]))
    summary = allocate_budget(summary, args.budget, args.min_share)
    summary.to_csv(outdir / "cluster_summary.csv", index=False)
    summary[["price_tier", "price_min", "price_max", "budget_share_pct", "budget_pkr"]].to_csv(
        outdir / "budget_plan.csv", index=False)

    brand_tier = (df.groupby(["brand", "price_tier"])
                    .agg(comments=("sentiment", "size"),
                         positive_pct=("sentiment", lambda s: round((s == "Positive").mean() * 100, 1)))
                    .reset_index())
    brand_tier.to_csv(outdir / "brand_tier_matrix.csv", index=False)

    best = summary.loc[summary["positive_pct"].idxmax()]
    print("\n" + summary.to_string(index=False))
    verdict = ("the difference between tiers is statistically significant" if p_value < 0.05
               else "the difference between tiers is NOT significant (could be chance)")
    print(f"\nChi-square (tier vs sentiment): chi2={chi2:.1f}, p={p_value:.4g} -> {verdict}")
    print(f"Most positive tier: {best['price_tier']} "
          f"(Rs {best['price_min']:,.0f}-{best['price_max']:,.0f}, {best['positive_pct']}% positive)")

    # Charts.
    tiers = summary["price_tier"].tolist()
    plot_price_clusters(items, tiers, outdir)
    plot_sentiment_by_tier(df, summary, outdir)
    plot_brand_tier_heatmap(brand_tier, tiers, args.min_comments, outdir)
    plot_budget(summary, args.budget, outdir)
    print(f"\nSaved tables and charts in '{outdir}/'")


if __name__ == "__main__":
    main()
