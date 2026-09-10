"""Collects news headlines from Google News RSS.

RSS is used instead of scraping the Google News web page: the RSS feed
is plain XML with a stable format, while the web page layout can change
at any time. Each article is tagged with the query that found it
(query_tag), so later we can look at sentiment per topic and not just
one big average.
"""

import logging
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import feedparser
import requests

from src.config import (
    GOOGLE_NEWS_PARAMS,
    GOOGLE_NEWS_RSS_URL,
    HTTP_BACKOFF_BASE,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    NEWS_QUERIES,
    USER_AGENT,
)
from src.storage.db import insert_news

logger = logging.getLogger(__name__)


def build_rss_url(query: str) -> str:
    params = {"q": query, **GOOGLE_NEWS_PARAMS}
    return f"{GOOGLE_NEWS_RSS_URL}?{urlencode(params)}"


def _fetch_raw(url: str) -> bytes:
    """GET the feed, with a timeout and a few retries."""
    last_err: Exception | None = None
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as err:
            last_err = err
            wait = HTTP_BACKOFF_BASE ** attempt
            logger.warning(
                "RSS attempt %d/%d failed (%s). Retrying in %.0fs.",
                attempt, HTTP_MAX_RETRIES, err, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Could not reach RSS feed: {url} ({last_err})")


def _parse_entries(raw: bytes, query_tag: str) -> list[dict]:
    """Parse the XML feed into a list of simple dicts ready for the database."""
    feed = feedparser.parse(raw)
    if feed.bozo:  # feed is not perfectly valid XML, but might still be usable
        logger.warning("Feed '%s' is partially malformed: %s",
                       query_tag, feed.bozo_exception)

    items: list[dict] = []
    for entry in feed.entries:
        title = getattr(entry, "title", "").strip()
        url = getattr(entry, "link", "").strip()
        if not title or not url:
            continue  # skip entries we cannot use

        published_utc = None
        parsed = getattr(entry, "published_parsed", None)
        if parsed:
            published_utc = datetime(
                *parsed[:6], tzinfo=timezone.utc
            ).isoformat(timespec="seconds")

        source = None
        if hasattr(entry, "source") and hasattr(entry.source, "title"):
            source = entry.source.title

        items.append(
            {
                "query_tag": query_tag,
                "title": title,
                "source": source,
                "url": url,
                "published_utc": published_utc,
            }
        )
    return items


def fetch_news_for_query(query_tag: str, query: str) -> list[dict]:
    url = build_rss_url(query)
    raw = _fetch_raw(url)
    items = _parse_entries(raw, query_tag)
    logger.info("Feed '%s': %d articles parsed", query_tag, len(items))
    return items


def run(queries: dict[str, str] | None = None) -> int:
    """Fetch every feed and save new articles. Returns how many are new.

    Each feed runs on its own, so if one fails, the others still work.
    """
    queries = queries or NEWS_QUERIES
    total_new = 0
    for tag, query in queries.items():
        try:
            items = fetch_news_for_query(tag, query)
            new = insert_news(items)
            total_new += new
            logger.info("Feed '%s': %d new articles inserted", tag, new)
        except Exception as err:
            logger.error("Feed '%s' skipped: %s", tag, err)
            continue
    return total_new
