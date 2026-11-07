# Khaadi Price Clustering & Marketing Budget Optimization

Project 2 of a brand-analytics series (Project 1: YouTube comment scraping and sentiment analysis for Khaadi).

**Research question:** Which price range of Khaadi products gets the most positive customer feedback, and how should a marketing budget be split across price tiers?

**Live dashboard:** [khaadipricetiers.streamlit.app](https://khaadipricetiers.streamlit.app/)

**Short answer from the current data:** the three price tiers show almost identical sentiment (55.0% / 56.5% / 56.9% positive), the differences are far from statistically significant (chi-square = 0.82, p = 0.935), and the sample is too small to detect a real difference. The budget split produced by the pipeline is therefore a **demonstration of the method, not a recommendation**.

## Table of contents

1. [Approach](#approach)
2. [Project structure](#project-structure)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Live dashboard (Streamlit)](#live-dashboard-streamlit)
6. [YouTube API quota](#youtube-api-quota)
7. [Data quality rules](#data-quality-rules)
8. [Dataset overview](#dataset-overview)
9. [Results](#results)
10. [Data issues found in this result set](#data-issues-found-in-this-result-set)
11. [Outputs](#outputs)
12. [Limitations](#limitations)
13. [Next steps](#next-steps)
14. [License](#license)

## Approach

1. **Collect** - find YouTube videos about single Khaadi products whose title or description states a price (`scrape_youtube_prices.py`). One video is treated as one product, and its comments as feedback on that product.
2. **Cluster** - group the products into price tiers (Budget / Mid-range / Premium) with KMeans on log-price; the number of tiers is chosen by silhouette score (`cluster_prices.py`).
3. **Analyse** - sentiment (VADER) per tier, with a chi-square test of whether the tiers really differ.
4. **Allocate** - split a marketing budget across tiers using a conservative positive-rate estimate (lower bound of the 95% Wilson interval) weighted by comment volume, with a minimum share per tier.

```
YouTube search -> video_candidates.csv -> comments.csv -> KMeans (log price) -> VADER sentiment
                                                                   |
                                         chi-square test <---------+---------> Wilson lower bound -> budget split
```

## Project structure

```
.
├── scrape_youtube_prices.py     # step 1: videos with a price -> comments CSV
├── cluster_prices.py            # step 2: clustering, sentiment, budget, charts
├── app.py                       # Streamlit dashboard (reads the CSVs below)
├── Khaadi_Price_Clustering.ipynb  # Colab notebook used to run the pipeline
├── requirements.txt
├── video_candidates.csv         # cached list of priced videos (262 rows)
├── item_clusters.csv            # one row per product with its tier (54 rows)
├── cluster_summary.csv          # one row per tier: price range, sentiment, budget
├── budget_plan.csv              # budget split per tier
├── brand_tier_matrix.csv        # brand x tier comments and positive %
├── price_clusters.png           # charts
├── sentiment_by_tier.png
├── brand_tier_heatmap.png
├── budget_allocation.png
└── README.md
```

> The result files are stored in the repository root so the dashboard can read them directly.
> When you run the scripts locally they write to `data/` and `output/`; `app.py` looks in
> `output/`, then `data/`, then the root, so either layout works.

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

> **Reproducing the results below:** the cheapest product kept in the result set costs PKR 1,500, so
> that run used a minimum price of 1,500 or lower. The example commands above use 2,000 and 3,000,
> which would drop the cheapest Budget items and change the tier boundaries.

## Live dashboard (Streamlit)

`app.py` is an interactive dashboard over the result files: tier summary, sentiment charts, a
price-vs-sentiment scatter, an adjustable budget split, and filterable product and video tables.
It only reads the CSVs, so it needs no API key.

**Run locally**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Deploy for free on Streamlit Community Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, choose this repository, branch `main`, main file `app.py`.
3. Click **Deploy**. After a few minutes you get a public link.

**Live app:** [khaadipricetiers.streamlit.app](https://khaadipricetiers.streamlit.app/)

> Free Streamlit apps go to sleep after a period of inactivity. If you see a "this app has gone to
> sleep" message, click **Yes, get this app back up** and wait a few seconds.

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

## Dataset overview

| Stage | Count | Notes |
|---|---|---|
| Candidate videos with a parsed price (`video_candidates.csv`) | 262 | 47 published in 2025, 215 in 2026 |
| ...of which flagged as promotional | 99 | excluded from the analysis |
| ...of which not promotional | 163 | eligible |
| Products analysed (non-promo videos with at least 1 comment) | **54** | 12 from 2025, 42 from 2026 |
| Comments analysed | **197** | 111 positive, 74 neutral, 12 negative (56.3% positive overall) |

Candidate prices range from PKR 500 to 32,425 (median about 2,625). The 54 analysed products
range from PKR 1,500 to 32,425 (median 3,700).

Comment volume is thin: the median product has 3 comments, and 18 of the 54 products (33%) have exactly one.

## Results

### 1. Price tiers (KMeans on log-price, k = 3)

| Tier | Price range (PKR) | Median price (PKR) | Distinct prices | Products | Comments | Comments / product |
|---|---|---|---|---|---|---|
| Budget | 1,500 - 2,750 | 1,975 | 13 | 16 | 40 | 2.5 |
| Mid-range | 3,000 - 8,000 | 3,700 | 10 | 21 | 92 | 4.4 |
| Premium | 17,712 - 32,425 | 32,425 | 3 | 17 | 65 | 3.8 |
| **Total** | 1,500 - 32,425 | 3,700 | 26 | **54** | **197** | 3.6 |

There are clear gaps between the tiers (2,750 -> 3,000 and 8,000 -> 17,712), but the Premium tier is
dominated by one price point; see [Data issues](#data-issues-found-in-this-result-set).

### 2. Sentiment per tier (VADER)

| Tier | Comments | Positive % | Neutral % | Negative % | Avg. compound score | 95% Wilson interval for positive % |
|---|---|---|---|---|---|---|
| Budget | 40 | 55.0 | 40.0 | 5.0 | 0.291 | 39.8 - 69.3 |
| Mid-range | 92 | 56.5 | 35.9 | 7.6 | 0.344 | 46.3 - 66.2 |
| Premium | 65 | 56.9 | 38.5 | 4.6 | 0.377 | 44.8 - 68.2 |

Raw counts (positive / neutral / negative): Budget 22 / 16 / 2, Mid-range 52 / 33 / 7, Premium 37 / 25 / 3.

The three confidence intervals overlap almost completely.

### 3. Do the tiers really differ? (chi-square test)

| Test | Chi-square | Degrees of freedom | p-value | Conclusion |
|---|---|---|---|---|
| Tier x (positive / neutral / negative) | 0.82 | 4 | 0.935 | no significant difference |
| Tier x (positive / not positive) | 0.04 | 2 | 0.98 | no significant difference |

**Interpretation.** Sentiment is statistically indistinguishable across tiers. With 40-92 comments per
tier the test can only detect very large differences, so this is **not evidence that price has no
effect** on sentiment. Roughly 400 comments per tier would be needed to detect a 10-point difference.

**The ranking is not stable.** The Mid-range tier's 56.5% includes one video about Khaadi's IPO (12
comments, 11 positive; see below). Without that video Mid-range falls to 51.2% positive (41 of 80), the
lowest of the three tiers. A ranking that flips after removing one video is noise, not a finding.

### 4. Budget allocation (total budget PKR 1,000,000)

| Tier | Price range (PKR) | Positive-rate lower bound (95%) | Budget share | Budget (PKR) |
|---|---|---|---|---|
| Budget | 1,500 - 2,750 | 39.8% | 25.2% | 252,000 |
| Mid-range | 3,000 - 8,000 | 46.3% | 40.7% | 407,000 |
| Premium | 17,712 - 32,425 | 44.8% | 34.0% | 340,000 |

(Shares are rounded to one decimal, so the amounts add up to 999,000. The exact weighting and the
minimum share per tier are set in `cluster_prices.py`.)

The pipeline gives Mid-range the largest share because it has both the highest conservative
positive-rate and the most comments, not because Mid-range customers are demonstrably happier. Since
the underlying rates are statistically indistinguishable, **treat this split as an illustration of the
method.** With more data the same code would produce a defensible split.

### 5. Brand x tier matrix

| Brand | Tier | Comments | Positive % |
|---|---|---|---|
| Khaadi | Budget | 40 | 55.0 |
| Khaadi | Mid-range | 92 | 56.5 |
| Khaadi | Premium | 65 | 56.9 |

Only one brand is in the data, so the matrix is a single row per tier. It becomes useful if the
scraper is run for competitors (Sapphire, Gul Ahmed, Sana Safinaz, ...).

### 6. Products with the most comments

| Rank | Product (video title, shortened) | Price (PKR) | Tier | Comments | Positive % |
|---|---|---|---|---|---|
| 1 | Khaadi IPO: From Customer to Owner at Rs 12.40? (**not a product**) | 6,200 | Mid-range | 12 | 91.7 |
| 2 | Khaadi New Summer Collection 2026 - Affordable Price Rs:3000 | 3,000 | Mid-range | 10 | 40.0 |
| 3 | Khaadi Beautiful Winter Collection 2026 - Rs:8000 | 8,000 | Mid-range | 9 | 33.3 |
| 3 | Khaadi Beautifull New Winter Collection 2025 - Rs 3500 | 3,500 | Mid-range | 9 | 44.4 |
| 3 | KHAADI KHAAS EID COLLECTION 2026 (Beautiful Eid Collection) | 32,425 | Premium | 9 | 44.4 |
| 6 | KHAADI Chapter 2 Eid Collection 2026 (New Eid Collection) | 32,425 | Premium | 8 | 50.0 |
| 6 | Khaadi Suiting by AR - Premium Shine Wash n Wear Winter | 3,500 | Mid-range | 8 | 62.5 |

## Data issues found in this result set

Reading `item_clusters.csv` and `video_candidates.csv` together shows several problems that matter
more than the choice of clustering algorithm. They should be fixed before drawing any business conclusion.

| # | Issue | Evidence | Effect |
|---|---|---|---|
| 1 | **A non-product video is in the data** | "Khaadi IPO ... at Rs 12.40?" is about the stock-market listing; the parsed price (6,200) is a share-price figure, not a product price. | It is the single largest source of comments (12) and 11 positive, inflating Mid-range. |
| 2 | **Premium is essentially one price point** | 14 of 17 Premium products have the identical price PKR 32,425 (the others: 17,712 x2 and 30,000 x1). Only 3 distinct prices in the tier. | The tier reflects one number repeated across many videos of one brand's Eid/Signature collections (likely a price from a description for a whole collection or a most-expensive item), so it may not be a real per-product price. |
| 3 | **Budget is largely resellers and look-alikes** | Titles include "Wholesale", "COD", "Multani Khaadi Susi" (a generic fabric name), "Bin Saeed", "Tawakkal & EGO", and "Khaadi & Sana Safinaz Rs.1600 Per Kg". | The Budget tier may describe unbranded or reseller content rather than genuine Khaadi products. |
| 4 | **Very few comments per product** | Median 3 comments; 18 of 54 products have only one (Budget 8, Mid-range 5, Premium 5). | Per-product rates are meaningless; only tier totals are usable, and those are small. |
| 5 | **Neutral is the biggest hidden category** | 36-40% of comments are neutral in every tier. | Likely many Roman Urdu comments that VADER cannot score (see Limitations). |
| 6 | **Previous README said "2024 or later"** | All analysed videos are from 2025 (12) or 2026 (42), matching the default `--since-year 2025`. | Corrected in this version. |

## Outputs

Generated by `cluster_prices.py` (in `output/` when run locally, in the repo root in this repository):

| File | Content |
|---|---|
| `item_clusters.csv` | 54 rows: brand, item (video title), price, comments, pos, neg, pos_pct, cluster_id, price_tier |
| `cluster_summary.csv` | 3 rows: price range, items, comments, positive / neutral / negative %, average score, Wilson lower bound, budget share and amount |
| `budget_plan.csv` | 3 rows: price range, budget share %, budget in PKR |
| `brand_tier_matrix.csv` | brand x tier: comments and positive % |
| `price_clusters.png` | the products plotted by price, coloured by tier |
| `sentiment_by_tier.png` | positive / neutral / negative share per tier |
| `brand_tier_heatmap.png` | brand x tier heatmap |
| `budget_allocation.png` | budget split per tier |

Input cache: `video_candidates.csv` (video_id, item, price, year, promo). The scraped `comments.csv`
(input of step 2) is not stored in this repository.

## Limitations

- Only videos that state a price are usable, which is a small and biased subset of all videos. Many are reseller, wholesale or sale videos, not official Khaadi content.
- Comments are about the video as much as about the product (presenter, video quality, seller).
- VADER is built for English and scores Roman Urdu comments as neutral; a multilingual model
  (e.g. XLM-RoBERTa) would be more accurate.
- With fewer than about 30 products per tier, tier statistics are only a preliminary estimate (here 16-21 products and 40-92 comments per tier).
- One video = one product is a strong assumption; titles like "Eid Collection" cover many items at different prices, so the stated price may not describe the item commenters discuss.
- Prices are nominal and span 2025-2026; inflation and sale seasons are not adjusted for.
- Positive sentiment does not necessarily mean higher sales or higher return on marketing spend.

## Next steps

1. **Remove non-product videos** (e.g. the IPO video) with a title blocklist (`ipo`, `stock`, `share`, `psx`) and re-run.
2. **Handle collection videos**: deduplicate repeated price points (PKR 32,425) or skip videos whose title says "collection" without a specific item.
3. **Filter or flag resellers** ("wholesale", "COD", "Multani Khaadi Susi", other brands in the title).
4. **Grow the sample** to roughly 400 comments per tier by running step 4 of the usage guide (`--only-extra`) over several days.
5. **Switch to a multilingual sentiment model** so Roman Urdu comments are scored instead of defaulting to neutral.
6. **Add competitor brands** to make the brand x tier matrix meaningful.
7. **Report uncertainty in the budget split** (e.g. bootstrap the shares) instead of a single set of numbers.

## License

MIT
