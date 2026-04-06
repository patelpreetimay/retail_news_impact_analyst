"""
news_scraper.py — Financial News Scraper for RNIA
=================================================

Collects financial news articles from multiple RSS feeds, extracts full
article text, removes duplicates, and saves the dataset as a CSV file.

RSS Sources (Global):
    - Yahoo Finance Top Stories / Market News / Industry
    - CNBC Top News / Markets
    - Investing.com News / Market Overview

RSS Sources (India):
    - Moneycontrol Latest / Markets / Business
    - Economic Times Markets / Corporate / Stocks
    - LiveMint Markets / Money
    - Hindu Business Line Markets / Stocks
    - Business Today
    - CNBCTV18 Market / Latest

Removed feeds (empirically zero or near-zero extractions):
    - Reuters / Bloomberg (Google-News-wrapped redirects can't be scraped)
    - Wall Street Journal, Financial Times (paywalled)
    - MarketWatch, Zee Business, Investing India (extraction failures)

Incremental Dataset:
    - Appends new articles to existing dataset
    - Deduplicates by URL
    - Maintains a rolling cap of MAX_DATASET_SIZE (10,000) rows
    - Trims oldest rows first when the cap is exceeded

Output:
    data/raw_news/news_raw_dataset.csv

Columns:
    headline, article_text, source, timestamp, url, region
"""

import os
import time
import logging
from datetime import datetime

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# RSS feed URLs — (source_name, feed_url) tuples
# Curated list — only feeds that empirically yield successful article
# extractions are kept. Feeds that consistently returned no text
# (Google-News-wrapped Reuters/Bloomberg, paywalled WSJ/FT,
# MarketWatch, Zee Business, Investing India) have been removed
# to avoid wasted scraping time.
RSS_FEEDS = [
    # ── Global Sources ─────────────────────────────────────────
    # Yahoo Finance
    ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
    ("Yahoo Finance", "https://finance.yahoo.com/rss/marketnews"),
    ("Yahoo Finance", "https://finance.yahoo.com/rss/industry"),
    # CNBC
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC", "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    # Investing.com
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ("Investing.com", "https://www.investing.com/rss/market_overview.rss"),

    # ── Indian Sources ─────────────────────────────────────────
    # Moneycontrol
    ("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/business.xml"),
    # Economic Times
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Economic Times", "https://economictimes.indiatimes.com/rssfeeds/13352306.cms"),
    ("Economic Times", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    # LiveMint
    ("LiveMint", "https://www.livemint.com/rss/markets"),
    ("LiveMint", "https://www.livemint.com/rss/money"),
    # Hindu Business Line
    ("Hindu BusinessLine", "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
    ("Hindu BusinessLine", "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss"),
    # Business Today
    ("Business Today", "https://www.businesstoday.in/rss/feed"),
    # CNBCTV18
    ("CNBCTV18", "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"),
    ("CNBCTV18", "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/latest.xml"),
]

# Indian source names — used for region tagging
INDIAN_SOURCES = {
    "Moneycontrol", "Economic Times", "LiveMint",
    "Hindu BusinessLine", "Business Today", "CNBCTV18",
}


def get_region(source_name: str) -> str:
    """Return 'India' if the source is Indian, else 'Global'."""
    return "India" if source_name in INDIAN_SOURCES else "Global"


# Maximum number of articles to collect
MAX_ARTICLES = 2000

# Maximum rolling dataset size — oldest rows are trimmed when exceeded
MAX_DATASET_SIZE = 10000

# Maximum share any single region may occupy (60 %). The minority region
# is always kept in full; the majority region is down-sampled if it
# exceeds this ratio so the model doesn't become biased.
MAX_REGION_RATIO = 0.60

# Output path (relative to the project root)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_news")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news_raw_dataset.csv")

# HTTP settings
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between requests (be polite to servers)
MAX_RETRIES = 1  # single attempt — retries on flaky sites just waste time
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def fetch_rss_entries(feed_url: str, source_name: str) -> list[dict]:
    """
    Parse an RSS feed and return a list of entry dicts.

    Each dict contains:
        - headline  (str): title of the article
        - url       (str): link to the full article
        - timestamp (str): publication date (ISO-like string)
        - source    (str): human-readable source name

    Parameters
    ----------
    feed_url : str
        URL of the RSS feed.
    source_name : str
        Human-readable name for the source (e.g. "Reuters").

    Returns
    -------
    list[dict]
        List of extracted entries.
    """
    logger.info("Fetching RSS feed: %s (%s)", source_name, feed_url)
    entries = []

    try:
        feed = feedparser.parse(feed_url)

        if feed.bozo:
            logger.warning(
                "Feed parser reported an issue for %s: %s",
                source_name,
                feed.bozo_exception,
            )

        for entry in feed.entries:
            # Extract the publication date
            published = entry.get("published", entry.get("updated", ""))
            # Try to normalise the date
            try:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6]).isoformat()
            except Exception:
                pass  # keep the raw string

            # Some feeds omit pubDate / updated entirely. Fall back to "now"
            # so downstream sorting (UI orders by timestamp DESC) doesn't
            # bury freshly-scraped items at the bottom.
            if not published or not str(published).strip():
                published = datetime.now().isoformat(timespec="seconds")

            entries.append(
                {
                    "headline": entry.get("title", "").strip(),
                    "url": entry.get("link", "").strip(),
                    "timestamp": published,
                    "source": source_name,
                }
            )

        logger.info("  → Found %d entries from %s", len(entries), source_name)

    except Exception as exc:
        logger.error("Failed to fetch feed %s: %s", source_name, exc)

    return entries


def fetch_article_text(url: str) -> str:
    """
    Download the full article page and extract the body text.

    Uses BeautifulSoup to pull text from <p> tags inside the article body.
    Falls back to all <p> tags if no article-specific container is found.

    Parameters
    ----------
    url : str
        URL of the article.

    Returns
    -------
    str
        Extracted article text, or an empty string on failure.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove unwanted tags that may pollute the text
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Attempt to locate the main article body
            article_body = (
                soup.find("article")
                or soup.find("div", class_="article-body")
                or soup.find("div", class_="caas-body")       # Yahoo Finance
                or soup.find("div", class_="ArticleBody")     # CNBC
                or soup.find("div", {"id": "article-body"})
                # Indian news sources
                or soup.find("div", class_="content_wrapper")  # Moneycontrol
                or soup.find("div", class_="artText")          # Economic Times
                or soup.find("div", class_="story-content")    # LiveMint
                or soup.find("div", class_="story_details")    # NDTV Profit
                or soup.find("div", class_="contentSec")       # Hindu BusinessLine
                or soup.find("div", class_="story-with-main-sec") # Business Today
                or soup.find("div", class_="article_content")  # CNBCTV18
                or soup.find("div", class_="article-content")  # Zee Business
            )

            if article_body:
                paragraphs = article_body.find_all("p")
            else:
                # Fallback: all <p> tags on the page
                paragraphs = soup.find_all("p")

            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                url,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RATE_LIMIT_DELAY * attempt)  # back-off

    return ""


def collect_news(max_articles: int = MAX_ARTICLES) -> pd.DataFrame:
    """
    Main collection routine.

    1. Fetches entries from all RSS feeds.
    2. De-duplicates entries by URL as it goes.
    3. Downloads and parses the full article text for each entry.
    4. Tags each article with a ``region`` (India / Global).
    5. Stops once *max_articles* unique articles have been collected.
    6. Returns a DataFrame capped at *max_articles* rows.

    Parameters
    ----------
    max_articles : int, optional
        Maximum number of articles to collect (default: 2000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: headline, article_text, source, timestamp, url, region.
    """
    all_entries: list[dict] = []
    seen_urls: set[str] = set()  # Track unique URLs across all feeds
    feeds_processed = 0
    total_raw_entries = 0
    duplicates_skipped = 0

    # Step 1 — Gather RSS entries from every feed, dedup as we go
    for source_name, feed_url in RSS_FEEDS:
        feeds_processed += 1
        entries = fetch_rss_entries(feed_url, source_name)
        total_raw_entries += len(entries)

        for entry in entries:
            url = entry["url"]
            if url in seen_urls:
                duplicates_skipped += 1
                continue
            seen_urls.add(url)
            all_entries.append(entry)

            # Stop early if we already have enough entries
            if len(all_entries) >= max_articles:
                break

        if len(all_entries) >= max_articles:
            logger.info("Reached target of %d entries. Stopping RSS fetch.", max_articles)
            break

    logger.info("Feeds processed       : %d / %d", feeds_processed, len(RSS_FEEDS))
    logger.info("Total raw RSS entries : %d", total_raw_entries)
    logger.info("Duplicates skipped    : %d", duplicates_skipped)
    logger.info("Unique entries to fetch: %d", len(all_entries))

    # Step 2 — Download full article text for each unique entry
    articles: list[dict] = []
    empty_count = 0

    for idx, entry in enumerate(all_entries, start=1):
        logger.info(
            "[%d/%d] Fetching article text: %s",
            idx,
            len(all_entries),
            entry["headline"][:80],
        )
        article_text = fetch_article_text(entry["url"])

        if not article_text:
            logger.warning("  ⚠ No text extracted for: %s", entry["url"])
            empty_count += 1
            continue  # Skip articles with no text

        articles.append(
            {
                "headline": entry["headline"],
                "article_text": article_text,
                "source": entry["source"],
                "timestamp": entry["timestamp"],
                "url": entry["url"],
                "region": get_region(entry["source"]),
            }
        )

        # Rate-limit to avoid overwhelming servers
        time.sleep(RATE_LIMIT_DELAY)

    # Step 3 — Build DataFrame
    df = pd.DataFrame(articles, columns=["headline", "article_text", "source", "timestamp", "url", "region"])

    # Final logging summary
    logger.info("-" * 40)
    logger.info("COLLECTION SUMMARY")
    logger.info("-" * 40)
    logger.info("Feeds processed        : %d", feeds_processed)
    logger.info("Total raw entries      : %d", total_raw_entries)
    logger.info("Duplicates removed     : %d", duplicates_skipped)
    logger.info("Empty articles dropped : %d", empty_count)
    logger.info("Final dataset size     : %d articles", len(df))

    return df


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    """
    Run the news collection pipeline and save results to CSV.

    Global deduplication behaviour:
        - If the output CSV already exists, new articles are **appended**.
        - **Primary dedup**: duplicates removed based on the ``url`` column.
        - **Fallback dedup**: for rows with missing URLs, duplicates are
          removed based on ``headline`` + ``timestamp``.
        - The dataset is sorted by ``timestamp`` (newest first).
        - If the total exceeds ``MAX_DATASET_SIZE`` (10,000), the oldest
          rows are trimmed so the final size equals ``MAX_DATASET_SIZE``.
    """
    logger.info("=" * 60)
    logger.info("RNIA — News Scraper Started")
    logger.info("=" * 60)

    # Collect new articles from RSS feeds
    df_new = collect_news(max_articles=MAX_ARTICLES)
    new_count = len(df_new)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 & 2 — Load existing dataset and combine with new data
    # ------------------------------------------------------------------
    existing_count = 0
    if os.path.isfile(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        existing_count = len(df_existing)
        logger.info("Existing dataset loaded: %d rows", existing_count)

        # Back-fill region column for legacy rows that lack it
        if "region" not in df_existing.columns:
            df_existing["region"] = df_existing["source"].apply(get_region)

        # Combine old + new
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        logger.info("No existing dataset found — starting fresh.")
        df_merged = df_new.copy()

    # Ensure every row has a region tag
    if "region" not in df_merged.columns:
        df_merged["region"] = df_merged["source"].apply(get_region)
    else:
        df_merged["region"] = df_merged["region"].fillna(
            df_merged["source"].apply(get_region)
        )

    combined_count = len(df_merged)

    # ------------------------------------------------------------------
    # STEP 3 — Global deduplication (by URL)
    # ------------------------------------------------------------------

    # Primary dedup: by URL (keep first / oldest occurrence)
    before_url_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset="url", keep="first")
    url_dupes_removed = before_url_dedup - len(df_merged)

    # Fallback dedup: for rows where URL may be missing/empty,
    # deduplicate by headline + timestamp combination
    before_fallback_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(
        subset=["headline", "timestamp"], keep="first"
    )
    fallback_dupes_removed = before_fallback_dedup - len(df_merged)

    total_dupes_removed = url_dupes_removed + fallback_dupes_removed

    # ------------------------------------------------------------------
    # STEP 4 — Global + India balance
    # ------------------------------------------------------------------
    # Prevent either region from dominating the dataset.
    # The minority region is kept in full; the majority is randomly
    # down-sampled so it does not exceed MAX_REGION_RATIO of the total.
    india_mask = df_merged["region"] == "India"
    n_india = india_mask.sum()
    n_global = len(df_merged) - n_india
    balance_trimmed = 0

    if len(df_merged) > 0:
        majority_is_global = n_global > n_india
        n_minority = min(n_india, n_global)
        n_majority = max(n_india, n_global)

        # Maximum majority size so that majority / total <= MAX_REGION_RATIO
        if n_minority > 0:
            max_majority = int(n_minority * MAX_REGION_RATIO / (1 - MAX_REGION_RATIO))
            if n_majority > max_majority:
                balance_trimmed = n_majority - max_majority
                if majority_is_global:
                    df_majority = df_merged[~india_mask].sample(n=max_majority, random_state=42)
                    df_merged = pd.concat([df_merged[india_mask], df_majority], ignore_index=True)
                else:
                    df_majority = df_merged[india_mask].sample(n=max_majority, random_state=42)
                    df_merged = pd.concat([df_merged[~india_mask], df_majority], ignore_index=True)

    # ------------------------------------------------------------------
    # STEP 5 — Sort by timestamp (newest first)
    # ------------------------------------------------------------------
    df_merged["timestamp"] = pd.to_datetime(
        df_merged["timestamp"], errors="coerce"
    )
    df_merged = df_merged.sort_values(
        by="timestamp", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # STEP 6 — Enforce rolling dataset size cap
    # ------------------------------------------------------------------
    rows_trimmed = 0
    if len(df_merged) > MAX_DATASET_SIZE:
        rows_trimmed = len(df_merged) - MAX_DATASET_SIZE
        df_merged = df_merged.head(MAX_DATASET_SIZE).reset_index(drop=True)

    final_count = len(df_merged)

    # Region counts for logging
    final_india = (df_merged["region"] == "India").sum()
    final_global = final_count - final_india

    # ------------------------------------------------------------------
    # STEP 7 — Save the final deduplicated, balanced, sorted dataset
    # ------------------------------------------------------------------
    df_merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # STEP 8 — Detailed logging summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCRAPER SUMMARY")
    print("=" * 60)
    print(f"  Existing dataset       : {existing_count}")
    print(f"  New scraped articles   : {new_count}")
    print(f"  Combined (before dedup): {combined_count}")
    print(f"  Duplicates removed     : {total_dupes_removed}")
    if url_dupes_removed:
        print(f"    - by URL             : {url_dupes_removed}")
    if fallback_dupes_removed:
        print(f"    - by headline+time   : {fallback_dupes_removed}")
    if balance_trimmed:
        print(f"  Balance trimmed        : {balance_trimmed}")
    if rows_trimmed:
        print(f"  Oldest rows trimmed    : {rows_trimmed}")
    print(f"  -------------------------------------")
    print(f"  Indian articles        : {final_india}")
    print(f"  Global articles        : {final_global}")
    print(f"  Total dataset size     : {final_count}")
    print(f"  Saved to               : {OUTPUT_FILE}")
    print("=" * 60)

    logger.info("Dataset saved to: %s", OUTPUT_FILE)
    logger.info("Final dataset size: %d articles (India=%d, Global=%d)",
                final_count, final_india, final_global)

    # Articles per source
    logger.info("-" * 40)
    logger.info("Articles per source:")
    for source, count in df_merged["source"].value_counts().items():
        logger.info("  %s: %d", source, count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

