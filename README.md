Khaadi Price Clustering & Marketing Budget Optimization
Project 2 of a brand-analytics series (Project 1: YouTube comment scraping and sentiment analysis for Khaadi).
Which price range of Khaadi products gets the most positive customer feedback, and how should a marketing budget be split across price tiers?
Approach
Collect - find YouTube videos about single Khaadi products whose title or description states a price (scrape_youtube_prices.py). One video is treated as one product, and its comments as feedback on that product.
Cluster - group the products into price tiers (Budget / Mid-range / Premium) with KMeans on log-price; the number of tiers is chosen by silhouette score (cluster_prices.py).
Analyse - sentiment (VADER) per tier, with a chi-square test of whether the tiers really differ.
Allocate - split a marketing budget across tiers using a conservative positive-rate estimate (lower bound of the 95% Wilson interval) weighted by comment volume, with a minimum share per tier.
Project structure
.
├── scrape_youtube_prices.py   # step 1: videos with a price -> comments CSV
├── cluster_prices.py          # step 2: clustering, sentiment, budget, charts
├── generate_demo_data.py      # synthetic data to test the pipeline
├── requirements.txt
└── README.md
Setup
pip install -r requirements.txt
You need a YouTube Data API v3 key (Google Cloud Console > APIs & Services > Credentials). Provide it as the YOUTUBE_API_KEY environment variable, a .env file, or a Colab Secret. Never commit the key.
Usage
# 1. See the quota cost first (no API call is made)
python scrape_youtube_prices.py --dry-run

# 2. Collect data (first run searches YouTube and caches the video list)
python scrape_youtube_prices.py --brand Khaadi --min-price 2000 --min-comments 1

# 3. Re-run with other filters without searching again
python scrape_youtube_prices.py --brand Khaadi --min-price 3000 --min-comments 1 --reuse

# 4. Cluster and allocate the budget
python cluster_prices.py --input data/comments.csv --budget 1000000 --k 3
Test the pipeline without any API key on synthetic data:
python generate_demo_data.py --out data/demo_comments.csv
python cluster_prices.py --input data/demo_comments.csv
YouTube API quota
The free quota is 10,000 units per day and resets at midnight Pacific Time (about 13:00 in Pakistan).
Call
search.list
videos.list
commentThreads.list
Cost
100 units
1 unit per 50 videos
1 unit per 100 comments
Used for
finding videos (the expensive part)
reading titles and descriptions
reading comments
A full search (30 queries x 2 pages) costs about 6,000 units, so it can be run once per day. The candidate videos are cached in data/video_candidates.csv; later runs with --reuse only spend a few hundred units on comments. --quota-budget (default 8,000) stops the script before the limit.
Data quality rules
Prices are parsed from the title first, then the description (Rs 4,500, PKR4500, 6,990/-).
Videos listing more than three different prices are treated as collections and skipped.
Titles such as "starting at Rs 2000", "sale", "70% off" are skipped: they quote an entry-level or promotional price, not the price of one product (--keep-promo disables this).
Videos published before --since-year (default 2025) are skipped, because older prices are not comparable after inflation.
--min-price removes very low prices that are usually sale or accessory prices.
Outputs (output/)
item_clusters.csv, cluster_summary.csv, budget_plan.csv, brand_tier_matrix.csv, price_clusters.png, sentiment_by_tier.png, brand_tier_heatmap.png, budget_allocation.png
Results
To be filled in after running on real data. Numbers produced by generate_demo_data.py are synthetic and must not be reported as findings.
Limitations
Only videos that state a price are usable, which is a small and biased subset of all videos.
Comments are about the video as much as about the product (presenter, video quality).
VADER is built for English and scores Roman Urdu comments as neutral; a multilingual model (e.g. XLM-RoBERTa) would be more accurate.
With fewer than about 30 products, tier statistics are only a preliminary estimate.
Positive sentiment does not necessarily mean higher sales.
License
MIT
