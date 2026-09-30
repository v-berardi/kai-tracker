"""Settings for the project: tickers, news queries, paths, network options.

Everything is here in one file, so changing a ticker or a query does not
mean editing the rest of the code.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "kai_tracker.db"

# --- Tickers we track ---
# Yahoo ticker -> readable name
TICKERS = {
    "NVDA": "Nvidia",
    "000660.KS": "SK Hynix",
    "005930.KS": "Samsung Electronics",  # also makes HBM, good for comparison
    "TSM": "TSMC",                        # makes the chips, part of the supply chain
}

# Price history fetched on first run
PRICE_HISTORY_PERIOD = "2y"
PRICE_INTERVAL = "1d"

# --- Google News RSS queries ---
# Each query is its own RSS feed, tagged with a topic name.
NEWS_QUERIES = {
    "sk_hynix_hbm": "SK Hynix HBM",
    "nvidia_supply": "Nvidia HBM supply",
    "samsung_hbm": "Samsung HBM3E",
    "memory_capex": "DRAM capex semiconductor",
    "korea_chips": "Korea semiconductor export",
    "tsmc_cowos": "TSMC CoWoS",
}

# --- News topics for each ticker ---
# The event study of a ticker only uses the news about this company (or
# its market). Without this, a Samsung headline could create an "event"
# that is then tested on the Nvidia price.
TICKER_TOPICS = {
    "NVDA": ["nvidia_supply"],
    "000660.KS": ["sk_hynix_hbm", "memory_capex", "korea_chips"],
    "005930.KS": ["samsung_hbm", "memory_capex", "korea_chips"],
    "TSM": ["tsmc_cowos"],
}

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
GOOGLE_NEWS_PARAMS = {"hl": "en-US", "gl": "US", "ceid": "US:en"}

# --- NLP / FinBERT ---
FINBERT_MODEL = "ProsusAI/finbert"
NLP_BATCH_SIZE = 16        # number of headlines scored per model call
NLP_MAX_LENGTH = 128       # headlines are short, so 128 tokens is enough

# --- Network ---
HTTP_TIMEOUT = 15          # seconds
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_BASE = 2.0    # wait time doubles each retry: 2s, 4s, 8s
USER_AGENT = (
    "KAI-Hardware-Tracker/0.1 (student research project; contact via GitHub)"
)
