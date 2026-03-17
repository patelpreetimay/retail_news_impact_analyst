"""
news_scraper.py — Financial News Scraper for RNIA
=================================================

Collects financial news articles from multiple RSS feeds, extracts full
article text, removes duplicates, and saves the dataset as a CSV file.

RSS Sources:
    - Reuters Business / Markets / Technology / Company News
    - Yahoo Finance Top Stories / Market News / Industry
    - CNBC Top News / Markets
    - MarketWatch Top Stories / Market Pulse
    - Investing.com News / Stock Market
    - Financial Times Home / Companies
    - Bloomberg (ETF Report)
    - Wall Street Journal Markets

Incremental Dataset:
    - Appends new articles to existing dataset
    - Deduplicates by URL
    - Maintains a rolling cap of MAX_DATASET_SIZE (10,000) rows
    - Trims oldest rows first when the cap is exceeded

Output:
    data/raw_news/news_raw_dataset.csv

Columns:
    headline, article_text, source, timestamp, url
"""

import os
import sys
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
# Multiple feeds per source for broader coverage toward 2000-article target.
RSS_FEEDS = [
    # Reuters
    ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/technologyNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/marketsNews"),
    ("Reuters", "https://feeds.reuters.com/reuters/companyNews"),
    # Yahoo Finance
    ("Yahoo Finance", "https://finance.yahoo.com/rss/topstories"),
    ("Yahoo Finance", "https://finance.yahoo.com/rss/marketnews"),
    ("Yahoo Finance", "https://finance.yahoo.com/rss/industry"),
    # CNBC
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC", "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    # MarketWatch
    ("MarketWatch", "https://www.marketwatch.com/rss/topstories"),
    ("MarketWatch", "https://www.marketwatch.com/rss/marketpulse"),
    # Investing.com
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ("Investing.com", "https://www.investing.com/rss/stock_market_news.rss"),
    # Financial Times
    ("Financial Times", "https://www.ft.com/rss/home"),
    ("Financial Times", "https://www.ft.com/rss/companies"),
    # Bloomberg
    ("Bloomberg", "https://www.bloomberg.com/feed/podcast/etf-report.xml"),
    # Wall Street Journal
    ("Wall Street Journal", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
]

# Maximum number of articles to collect
MAX_ARTICLES = 2000

# Maximum rolling dataset size — oldest rows are trimmed when exceeded
MAX_DATASET_SIZE = 10000

# Output path (relative to the project root)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_news")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news_raw_dataset.csv")

# HTTP settings
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between requests (be polite to servers)
MAX_RETRIES = 3
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

    1. Fetches entries from all RSS feeds (15 feeds across 6 sources).
    2. De-duplicates entries by URL as it goes.
    3. Downloads and parses the full article text for each entry.
    4. Stops once *max_articles* unique articles have been collected.
    5. Returns a DataFrame capped at *max_articles* rows.

    Parameters
    ----------
    max_articles : int, optional
        Maximum number of articles to collect (default: 2000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: headline, article_text, source, timestamp, url.
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
            }
        )

        # Rate-limit to avoid overwhelming servers
        time.sleep(RATE_LIMIT_DELAY)

    # Step 3 — Build DataFrame
    df = pd.DataFrame(articles, columns=["headline", "article_text", "source", "timestamp", "url"])

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

        # Combine old + new
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        logger.info("No existing dataset found — starting fresh.")
        df_merged = df_new.copy()

    combined_count = len(df_merged)

    # ------------------------------------------------------------------
    # STEP 3 — Global deduplication
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
    # STEP 4 — Sort by timestamp (newest first)
    # ------------------------------------------------------------------
    df_merged["timestamp"] = pd.to_datetime(
        df_merged["timestamp"], errors="coerce"
    )
    df_merged = df_merged.sort_values(
        by="timestamp", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # STEP 5 — Enforce rolling dataset size cap
    # ------------------------------------------------------------------
    rows_trimmed = 0
    if len(df_merged) > MAX_DATASET_SIZE:
        rows_trimmed = len(df_merged) - MAX_DATASET_SIZE
        df_merged = df_merged.head(MAX_DATASET_SIZE).reset_index(drop=True)

    final_count = len(df_merged)

    # ------------------------------------------------------------------
    # STEP 6 — Save the final deduplicated, sorted dataset
    # ------------------------------------------------------------------
    df_merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # STEP 7 — Detailed logging summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GLOBAL DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"  Existing dataset       : {existing_count}")
    print(f"  New scraped articles   : {new_count}")
    print(f"  Combined (before dedup): {combined_count}")
    print(f"  Duplicates removed     : {total_dupes_removed}")
    if url_dupes_removed:
        print(f"    ├─ by URL            : {url_dupes_removed}")
    if fallback_dupes_removed:
        print(f"    └─ by headline+time  : {fallback_dupes_removed}")
    if rows_trimmed:
        print(f"  Oldest rows trimmed    : {rows_trimmed}")
    print(f"  Final dataset size     : {final_count}")
    print(f"  Saved to               : {OUTPUT_FILE}")
    print("=" * 60)

    logger.info("Dataset saved to: %s", OUTPUT_FILE)
    logger.info("Final dataset size: %d articles", final_count)

    # Articles per source
    logger.info("-" * 40)
    logger.info("Articles per source:")
    for source, count in df_merged["source"].value_counts().items():
        logger.info("  %s: %d", source, count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

