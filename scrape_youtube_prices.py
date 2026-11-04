"""Collect YouTube comments for videos that state a product price.

Idea: one video == one product. The price is parsed from the video title or
description, and the video's comments are treated as customer feedback about
that product.

Output CSV columns (this is the input of ``cluster_prices.py``):
    brand, item, price, comment, likes, video_id

YouTube Data API v3 quota
-------------------------
The default quota is 10,000 units per day and resets at midnight Pacific
Time (about 13:00 in Pakistan). Costs used by this script:

    search.list           100 units per call  (the expensive one)
    videos.list             1 unit  per call  (up to 50 videos)
    commentThreads.list     1 unit  per call  (up to 100 comments)

A full run (30 queries x 2 pages) costs about 6,000 units for searching plus a
few hundred for comments. Running the full search twice in one day exhausts
the quota. The list of candidate videos is therefore cached to
``video_candidates.csv`` and can be re-used with ``--reuse`` for about
200-300 units.

Usage
-----
    python scrape_youtube_prices.py --dry-run                  # cost estimate only
    python scrape_youtube_prices.py --brand Khaadi --min-price 2000
    python scrape_youtube_prices.py --brand Khaadi --min-price 3000 --reuse
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from googleapiclient.errors import HttpError
except ImportError:  # lets the module be imported (e.g. for tests) without the client library
    HttpError = Exception

log = logging.getLogger("scraper")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# API cost per call, in quota units.
SEARCH_COST = 100
LIST_COST = 1
COMMENTS_COST = 1
DAILY_QUOTA = 10_000

# Prices outside this range are treated as noise (years, phone numbers, ...).
MIN_PLAUSIBLE_PRICE = 500
MAX_PLAUSIBLE_PRICE = 100_000

# "Rs 4,500", "PKR4500", "4500/-", "4,500 rs" ...
PRICE_RE = re.compile(
    r"(?:rs\.?|pkr|rupees?)\s*[:=\-]?\s*(\d{1,3}(?:,\d{3})+|\d{3,6})"
    r"|(\d{1,3}(?:,\d{3})+|\d{3,6})\s*(?:/-|rs\b|rs\.|pkr\b|rupees?\b)",
    re.IGNORECASE,
)

# Titles like "starting at Rs 2000", "70% off", "sale" quote a promotional or
# entry-level price, not the price of one product.
PROMO_RE = re.compile(
    r"start(?:s|ing)?\b|from rs|onwards?|up\s?to|upto|\bsale\b|\boff\b|%|discount|"
    r"flat|clearance|only rs|under rs|budget",
    re.IGNORECASE,
)

SEARCH_TOPICS = [
    "lawn suit", "unstitched", "ready to wear", "winter collection", "khaddar",
    "kurta", "unboxing", "review", "luxury", "sale", "new arrival", "pret",
    "3 piece", "2 piece", "dupatta", "shawl", "embroidered", "printed", "festive",
    "eid collection", "wedding", "casual", "formal", "velvet", "cambric", "linen",
    "haul", "outlet", "price", "price in pakistan",
]


# --------------------------------------------------------------------------- #
# Quota accounting
# --------------------------------------------------------------------------- #
@dataclass
class QuotaTracker:
    """Counts the quota units this run has spent and enforces a budget."""

    budget: int
    used: int = 0
    exhausted: bool = False  # set when the API itself reports quotaExceeded

    def can_spend(self, cost: int) -> bool:
        return not self.exhausted and self.used + cost <= self.budget

    def spend(self, cost: int) -> None:
        self.used += cost


def is_quota_error(err: Exception) -> bool:
    return "quota" in str(err).lower()


# --------------------------------------------------------------------------- #
# Price parsing
# --------------------------------------------------------------------------- #
def extract_prices(text: str) -> list[int]:
    """Return every plausible rupee amount found in ``text``."""
    prices = []
    for match in PRICE_RE.finditer(text or ""):
        value = int((match.group(1) or match.group(2)).replace(",", ""))
        if MIN_PLAUSIBLE_PRICE <= value <= MAX_PLAUSIBLE_PRICE:
            prices.append(value)
    return prices


def video_price(title: str, description: str, max_distinct: int = 3) -> int | None:
    """Price of the product shown in a video, or None if it cannot be trusted.

    The title is checked first, then the description. A text that lists more
    than ``max_distinct`` different prices describes a collection, not a single
    product, so it is rejected.
    """
    for text in (title, description):
        prices = extract_prices(text)
        if prices:
            distinct = sorted(set(prices))
            if len(distinct) > max_distinct:
                return None
            return int(statistics.median(distinct))
    return None


# --------------------------------------------------------------------------- #
# YouTube API stages
# --------------------------------------------------------------------------- #
def build_queries(brand: str) -> list[str]:
    return [f"{brand} {topic} price rs" for topic in SEARCH_TOPICS]


def search_video_ids(yt, queries: list[str], pages: int, page_size: int,
                     tracker: QuotaTracker) -> list[str]:
    """Stage 1: run the search queries and collect unique video ids."""
    found: dict[str, None] = {}
    for query in queries:
        token = None
        for _ in range(pages):
            if not tracker.can_spend(SEARCH_COST):
                log.warning("Quota budget reached (%d units); stopping the search.", tracker.used)
                return list(found)
            try:
                response = yt.search().list(
                    q=query, part="id", type="video",
                    pageToken=token, maxResults=min(50, page_size),
                ).execute()
            except HttpError as err:
                if is_quota_error(err):
                    tracker.exhausted = True
                    log.error("YouTube daily quota exhausted during search.")
                    return list(found)
                log.warning("Search failed for %r: %s", query, err)
                break
            tracker.spend(SEARCH_COST)
            for item in response.get("items", []):
                found[item["id"]["videoId"]] = None
            token = response.get("nextPageToken")
            if not token:
                break
    return list(found)


def fetch_candidates(yt, video_ids: list[str], brand: str,
                     tracker: QuotaTracker) -> list[dict]:
    """Stage 2: read title/description of each video and keep those with a price."""
    candidates = []
    for start in range(0, len(video_ids), 50):
        if not tracker.can_spend(LIST_COST):
            log.warning("Quota budget reached; stopping video detail lookup.")
            break
        batch = video_ids[start:start + 50]
        try:
            response = yt.videos().list(part="snippet", id=",".join(batch)).execute()
        except HttpError as err:
            if is_quota_error(err):
                tracker.exhausted = True
                log.error("YouTube daily quota exhausted while reading video details.")
                break
            log.warning("videos.list failed: %s", err)
            continue
        tracker.spend(LIST_COST)

        for item in response.get("items", []):
            snippet = item["snippet"]
            title = html.unescape(snippet["title"])
            description = snippet.get("description", "")
            if brand.lower() not in f"{title} {description}".lower():
                continue  # unrelated video or a different brand
            price = video_price(title, description)
            if price is None:
                continue
            published = snippet.get("publishedAt", "")
            candidates.append({
                "video_id": item["id"],
                "item": title,
                "price": price,
                "year": int(published[:4]) if published[:4].isdigit() else 0,
                "promo": bool(PROMO_RE.search(title)),
            })
    return candidates


def filter_candidates(candidates: list[dict], min_price: int, since_year: int,
                      keep_promo: bool) -> list[dict]:
    """Apply the quality filters and report how many videos each one removes."""
    log.info("Videos with a price found: %d", len(candidates))
    kept = [c for c in candidates if c["price"] >= min_price]
    log.info("  after min-price %d: %d", min_price, len(kept))
    # Older prices are not comparable because of inflation.
    kept = [c for c in kept if c.get("year", 0) >= since_year]
    log.info("  published in %d or later: %d", since_year, len(kept))
    if not keep_promo:
        kept = [c for c in kept if not c.get("promo", False)]
        log.info("  without 'starting at' / sale / off titles: %d", len(kept))
    return kept


def fetch_comments(yt, video: dict, brand: str, max_comments: int,
                   tracker: QuotaTracker) -> tuple[list[dict], str]:
    """Stage 3: fetch top-level comments of one video.

    Returns (rows, status) where status is one of
    "ok", "closed" (comments disabled), "quota" or "budget".
    """
    rows: list[dict] = []
    token = None
    while len(rows) < max_comments:
        if not tracker.can_spend(COMMENTS_COST):
            return rows, "budget"
        try:
            response = yt.commentThreads().list(
                part="snippet", videoId=video["video_id"], maxResults=100,
                pageToken=token, textFormat="plainText", order="relevance",
            ).execute()
        except HttpError as err:
            if is_quota_error(err):
                tracker.exhausted = True
                return rows, "quota"
            return rows, "closed"
        tracker.spend(COMMENTS_COST)

        for thread in response.get("items", []):
            snippet = thread["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "brand": brand, "item": video["item"], "price": video["price"],
                "comment": snippet["textDisplay"], "likes": snippet["likeCount"],
                "video_id": video["video_id"],
            })
        token = response.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)  # be polite to the API
    return rows[:max_comments], "ok"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_api_key() -> str | None:
    """Read the key from the environment, a .env file, or Colab Secrets."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("YOUTUBE_API_KEY")
    if key:
        return key
    try:  # Colab: left sidebar > key icon > add a secret named YOUTUBE_API_KEY
        from google.colab import userdata
        return userdata.get("YOUTUBE_API_KEY")
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--brand", default="Khaadi", help="brand name used in searches and filtering")
    p.add_argument("--per-query", type=int, default=50, help="videos per search page (max 50)")
    p.add_argument("--pages", type=int, default=2, help="search pages per query")
    p.add_argument("--max-comments", type=int, default=100, help="comments kept per video")
    p.add_argument("--min-comments", type=int, default=3, help="drop videos with fewer comments")
    p.add_argument("--min-price", type=int, default=MIN_PLAUSIBLE_PRICE,
                   help="drop videos priced below this (removes sale / entry-level prices)")
    p.add_argument("--since-year", type=int, default=2025,
                   help="drop videos published before this year (inflation)")
    p.add_argument("--keep-promo", action="store_true",
                   help="keep 'starting at' / sale / off videos (dropped by default)")
    p.add_argument("--quota-budget", type=int, default=8_000,
                   help="stop before spending more than this many quota units")
    p.add_argument("--reuse", action="store_true",
                   help="reuse the cached video list instead of searching again")
    p.add_argument("--dry-run", action="store_true", help="print the quota estimate and exit")
    p.add_argument("--out", default="data/comments.csv")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    out_path = Path(args.out)
    cache_path = out_path.parent / "video_candidates.csv"
    use_cache = args.reuse and cache_path.exists()
    queries = build_queries(args.brand)

    # Quota estimate (comments cost ~1 unit per video, so this is an upper-bound guess).
    search_units = 0 if use_cache else len(queries) * args.pages * SEARCH_COST
    print(f"Estimated quota: search {search_units:,} + details/comments ~300 "
          f"= ~{search_units + 300:,} of {DAILY_QUOTA:,} units/day "
          f"(budget guard: {args.quota_budget:,}).")
    if not use_cache and args.reuse:
        print(f"Note: {cache_path} not found, a fresh search will run.")
    if args.dry_run:
        return

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("YOUTUBE_API_KEY not found (environment, .env file or Colab Secrets).")
    from googleapiclient.discovery import build
    yt = build("youtube", "v3", developerKey=api_key)
    tracker = QuotaTracker(budget=args.quota_budget)

    # Stage 1 + 2: find candidate videos (or load them from the cache).
    if use_cache:
        log.info("Loading cached video list: %s", cache_path)
        candidates = pd.read_csv(cache_path).to_dict("records")
    else:
        video_ids = search_video_ids(yt, queries, args.pages, args.per_query, tracker)
        log.info("Unique videos found: %d", len(video_ids))
        candidates = fetch_candidates(yt, video_ids, args.brand, tracker)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candidates).to_csv(cache_path, index=False, encoding="utf-8-sig")
        log.info("Cached %d candidate videos -> %s", len(candidates), cache_path)

    videos = filter_candidates(candidates, args.min_price, args.since_year, args.keep_promo)

    # Stage 3: comments.
    rows: list[dict] = []
    stats = {"closed": 0, "too_few": 0}
    for video in videos:
        video_rows, status = fetch_comments(yt, video, args.brand, args.max_comments, tracker)
        if status in ("quota", "budget"):
            break  # no point continuing
        if status == "closed":
            stats["closed"] += 1
        elif len(video_rows) < args.min_comments:
            stats["too_few"] += 1
        else:
            rows += video_rows

    log.info("Videos with comments disabled: %d | with fewer than %d comments: %d",
             stats["closed"], args.min_comments, stats["too_few"])
    if tracker.exhausted:
        log.error("YouTube QUOTA EXHAUSTED: remaining videos were skipped. The quota resets "
                  "around 13:00 Pakistan time; re-run with --reuse then.")
    elif tracker.used >= args.quota_budget:
        log.warning("Quota budget (%d) reached: remaining videos were skipped.", args.quota_budget)
    log.info("Quota units used by this run: %d", tracker.used)

    if not rows:
        raise SystemExit("No comments collected.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["video_id", "comment"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Done: {df['item'].nunique()} items, {len(df)} comments -> {out_path}")


if __name__ == "__main__":
    main()
