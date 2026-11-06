# Khaadi Price Clustering & Marketing Budget Optimization

Project 2 of a brand-analytics series (Project 1: YouTube comment scraping and sentiment analysis for Khaadi).

Which price range of Khaadi products gets the most positive customer feedback, and how should a marketing budget be split across price tiers?

## Approach

1. **Collect** - find YouTube videos about single Khaadi products whose title or description states a price (`scrape_youtube_prices.py`). One video is treated as one product, and its comments as feedback on that product.
2. **Cluster** - group the products into price tiers (Budget / Mid-range / Premium) with KMeans on log-price; the number of tiers is chosen by silhouette score (`cluster_prices.py`).
3. **Analyse** - sentiment (VADER) per tier, with a chi-square test of whether the tiers really differ.
4. **Allocate** - split a marketing budget across tiers using a conservative positive-rate estimate (lower bound of the 95% Wilson interval) weighted by comment volume, with a minimum share per tier.

## Project structure

```
.
├── scrape_youtube_prices.py   # step 1: videos with a price -> comments CSV
├── cluster_prices.py          # step 2: clustering, sentiment, budget, charts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

You need a YouTube Data API v3 key (Google Cloud Console > APIs & Services > Credentials).
Provide it as the `YOUTUBE_API_KEY` environment variable, a `.env` file, or a Colab Secret.
Never commit the key.

## Usage

```bash
# 1. See the quota cost first (no API call is made)
python scrape_youtube_prices.py --dry-run

# 2. Collect data (first run searches YouTube and caches the video list)
python scrape_youtube_prices.py --brand Khaadi --min-price 2000 --min-comments 1

# 3. Re-run with other filters without searching again
python scrape_youtube_prices.py --brand Khaadi --min-price 3000 --min-comments 1 --reuse

# 4. Grow the dataset on another day: search only new keywords. New videos are
#    merged into the cached list, so the dataset accumulates over several days.
python scrape_youtube_prices.py --brand Khaadi --only-extra \
    --extra-topics "haul,suit price kitni,winter 2026,new collection,try on" \
    --min-price 2000 --min-comments 1

# 5. Cluster and allocate the budget
python cluster_prices.py --input data/comments.csv --budget 1000000 --k 3
```


## YouTube API quota

The free quota is 10,000 units per day and resets at midnight Pacific Time (about 13:00 in Pakistan).

| Call | Cost | Used for |
|---|---|---|
| `search.list` | 100 units | finding videos (the expensive part) |
| `videos.list` | 1 unit per 50 videos | reading titles and descriptions |
| `commentThreads.list` | 1 unit per 100 comments | reading comments |

A full search (30 queries x 2 pages) costs about 6,000 units, so it can be run **once per day**.
The candidate videos are cached in `data/video_candidates.csv`; later runs with `--reuse` only spend
a few hundred units on comments. `--quota-budget` (default 8,000) stops the script before the limit.

## Data quality rules

- Prices are parsed from the title first, then the description (`Rs 4,500`, `PKR4500`, `6,990/-`).
- Videos listing more than three different prices are treated as collections and skipped.
- Titles such as "starting at Rs 2000", "sale", "70% off" are skipped: they quote an entry-level or
  promotional price, not the price of one product (`--keep-promo` disables this).
- Videos published before `--since-year` (default 2025) are skipped, because older prices are not
  comparable after inflation.
- `--min-price` removes very low prices that are usually sale or accessory prices.

## Outputs (`output/`)

`item_clusters.csv`, `cluster_summary.csv`, `budget_plan.csv`, `brand_tier_matrix.csv`,
`price_clusters.png`, `sentiment_by_tier.png`, `brand_tier_heatmap.png`, `budget_allocation.png`

## Results

**Preliminary result (54 products, 197 comments, videos published 2024 or later):**
no statistically significant difference in sentiment across price tiers
(chi-square = 0.8, p = 0.94).

| Tier | Price range (PKR) | Products | Comments |
|---|---|---|---|
| Budget | 2,000 - 2,750 | 16 | 40 |
| Mid-range | up to 8,000 | 21 | 92 |
| Premium | 17,712 - 32,425 | 17 | 65 |

Full per-tier figures are in `output/cluster_summary.csv`.

The sample is too small (about 4 comments per product, 40-90 comments per tier) to detect
moderate differences, so this is **not** evidence that price has no effect on sentiment. Which
tier looked "most positive" changed between runs with different data, which is a sign that
the differences are noise at this sample size. Roughly 400 comments per tier would be needed to
detect a 10-point difference. The budget split in `output/budget_plan.csv` is derived from
these unreliable rates and should be treated as a demonstration of the method, not as a
recommendation.

## Limitations

- Only videos that state a price are usable, which is a small and biased subset of all videos.
- Comments are about the video as much as about the product (presenter, video quality).
- VADER is built for English and scores Roman Urdu comments as neutral; a multilingual model
  (e.g. XLM-RoBERTa) would be more accurate.
- With fewer than about 30 products, tier statistics are only a preliminary estimate.
- Positive sentiment does not necessarily mean higher sales.

## License

MIT
