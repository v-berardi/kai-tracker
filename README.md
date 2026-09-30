# K-AI Hardware Tracker

I built this project to follow the news around the AI memory chip
supply chain (SK Hynix, Nvidia, Samsung, TSMC) and compare it with
their stock prices. It collects news and stock prices, scores each
news headline with a financial NLP model (FinBERT), and then checks
with a basic statistical test if sentiment spikes are actually
followed by unusual stock returns.

**This is a monitoring tool, not a trading tool.** It does not try to
predict prices. It checks if there is a link between news sentiment
and stock returns, and shows the real result, even when that result is
"no clear link" or something that doesn't fit a simple story.

## Why I built this

I wanted a project that goes through a full, real pipeline: collecting
data, storing it, running an NLP model on it, and checking the result
with an actual statistical test instead of just assuming sentiment and
price are related. It also let me practice building a small dashboard
on top of it.

## The pipeline

```
scripts/run_ingestion.py   -->  prices + market indexes (yfinance) + news (Google News RSS)
scripts/run_sentiment.py   -->  scores new articles with FinBERT
scripts/run_event_study.py -->  checks if sentiment spikes line up with abnormal returns
app/dashboard.py           -->  Streamlit dashboard showing all of the above
```

Project layout:

```
kai-tracker/
├── data/                  # SQLite database (created on first run, not in git)
├── scripts/
│   ├── run_ingestion.py    # collect prices + news
│   ├── run_sentiment.py    # score articles with FinBERT
│   └── run_event_study.py  # print the event study results
├── src/
│   ├── config.py            # tickers, news queries, settings
│   ├── ingestion/
│   │   ├── market.py         # stock prices from yfinance
│   │   └── news.py           # news from Google News RSS
│   ├── nlp/
│   │   └── sentiment.py      # FinBERT sentiment scoring
│   ├── storage/
│   │   └── db.py             # SQLite database functions
│   └── analysis/
│       └── event_study.py    # the statistical test
├── tests/
│   └── test_event_study.py   # unit tests, run without a database
└── app/
    └── dashboard.py          # Streamlit + Plotly dashboard
```

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_ingestion.py             # 1. collect prices + news
python scripts/run_sentiment.py --limit 20  # 2. score a small batch first
python scripts/run_sentiment.py             #    then score the rest
python scripts/run_event_study.py --all-tickers  # 3. results for the 4 tickers
streamlit run app/dashboard.py              # 4. open the dashboard
```

The first time `run_sentiment.py` runs, it downloads FinBERT (about
440MB, then it stays cached). Scoring is incremental: only new
articles get scored each time you run it again.

You can run the scripts more than once without breaking anything:
prices get updated in place and articles are never added twice
(checked by URL).

### Running the tests

```bash
pytest tests/ -v
```

The tests use made-up sentiment and price numbers, so they check the
event study logic (event detection, CAR, CAAR, the t-test, overlapping
events) without needing the real database. The database tests use a
temporary SQLite file, so my real database is not touched.

## What works

- Collecting prices for 4 tickers and news for 6 topics runs
  end-to-end and fills the database.
- FinBERT scoring is incremental: only new articles get scored each
  time.
- The event study runs on real data and gives a result I can trust,
  because I checked the logic with unit tests on made-up numbers where
  I know the correct answer in advance.
- The dashboard shows price vs. sentiment and the event study results,
  and stays readable even with almost two years of daily data.

## What's limited

- **Google News RSS is one, English-only source.** It probably misses
  a lot of Korean coverage, which likely matters a lot for SK Hynix and
  Samsung specifically.
- **FinBERT only scores the headline, not the full article**, to keep
  ingestion fast. This loses some nuance a full-text model would catch.
- **The market index contains the stock itself.** Nvidia and TSMC are
  big parts of the SOXX ETF, and Samsung and SK Hynix are big parts of
  the KOSPI. So when Nvidia moves, the index also moves a bit because
  of Nvidia, and the market model removes a part of Nvidia's own
  abnormal return. A cleaner benchmark would be the index without the
  stock, but I don't have this data for free.

## Example result, and why I'm not hiding it

> **Note:** this output comes from an older version of the study,
> before I fixed the news topics per ticker, the time zones, the
> overlapping events and the before/after split of the CAR. I will
> replace it with the output of the new version.

This is the real output of `python scripts/run_event_study.py --ticker NVDA`,
on about 2 years of data (47 sentiment spikes detected):

```
[all     ]  n=44  CAAR=-0.0132  t=-1.44  p=0.158  -> not significant
[positive]  n=36  CAAR=-0.0227  t=-2.33  p=0.026  -> significant
[negative]  n= 8  CAAR=+0.0293  t=+1.47  p=0.185  -> not significant
```

The overall test ("all") is not significant, which is what I expected
going in. The "positive" sub-test alone does come out significant at
p < 0.05 — but with a *negative* CAAR, meaning positive sentiment spikes
were actually followed by slightly worse returns, not better ones. I'm
not reading this as "sell the good news": I ran three sub-tests here
(all / positive / negative), and with a 5% significance level, getting
one result like this by chance alone is not surprising. Treating it as
a real, tradeable effect without testing it on more data or a different
period would be a mistake.

I'm showing this instead of only the "clean" not-significant numbers
because that's the actual point of running a statistical test: to see
what the data says, including the part that doesn't fit a simple
story, instead of only keeping the result that looks good.

## Design choices

- **Every row stores when it was last written** (`ingested_at_utc`).
  For news it is when I collected the article. For prices it is the
  last update, because prices are rewritten at each run (yfinance
  changes old adjusted prices after splits and dividends). I use it
  only to check the data, the event study doesn't use it.
- **Each ticker uses only its own news topics** (`TICKER_TOPICS` in
  `src/config.py`). At first all the topics were mixed together, so a
  headline about Samsung could create an event that was then tested on
  Nvidia's price. Now the Nvidia study only uses the Nvidia topic.
- **News is matched to the right trading day.** News times are in UTC,
  but Nvidia trades in New York and SK Hynix in Seoul. I convert each
  headline to the local time of the exchange, and if it comes out after
  the market close, it counts for the next trading day (`MARKET_HOURS`
  in `src/config.py`).
- **RSS instead of scraping the news website**: the RSS feed is simple,
  structured XML, and doesn't change format as often as a web page.
- **Collecting data and scoring it are two separate steps.** New
  articles get a `sentiment_score` of NULL first, and a second script
  fills it in later. Collecting news never has to wait for the NLP
  model to load.
- **SQLite**: no server to run, and a UNIQUE constraint on the article
  URL gives me deduplication for free.
- **Each news feed runs on its own**, so if one of them fails, the
  others still work.
- **The CAR is split in "before" [-5, -1] and "after" [0, +5].** My
  question is if returns move *after* the news, so the "after" part is
  the main test. The "before" part is still useful: a lot of headlines
  talk about a price move that already happened ("Nvidia shares jump"),
  so a big "before" CAR means the news follows the price and not the
  opposite.
- **Market model for the normal return.** At first I used the
  constant mean model (normal return = average return of the stock
  before the event). On Nvidia it gave a strange result: in several
  events the normal return was negative, because the stock was going
  down in the estimation window, so any rebound later looked
  "abnormal". Now I fit `stock return = alpha + beta * index return`
  on the estimation window, with SOXX (US chip stocks) for Nvidia and
  TSMC and the KOSPI for Samsung and SK Hynix (`BENCHMARKS` in
  `src/config.py`). This way, a day where the whole market goes up is
  not counted as abnormal. The old model is still there with
  `--model constant`, to compare.
- **All tickers are reported together.** `--all-tickers` runs the study
  on the 4 stocks and prints one table, with the number of tests and a
  Bonferroni correction at the end. I report all of them, because
  keeping only the ticker with the best p-value would be p-hacking.
- **±5 trading day event window**: about a week on each side, long
  enough to see if the market keeps reacting, short enough that an
  unrelated news story is unlikely to land in the same window.
- **Events do not overlap.** The t-test supposes that the events are
  independent. If two event windows share some days, the same returns
  are counted two times and the test looks more significant than it
  really is. So I skip an event if it starts inside the window of the
  previous one, and I remove the event window days from the "normal
  return" of the other events.

## License

MIT — see [LICENSE](LICENSE).

## Author

Vincent Berardi, data science master's student at EURECOM. Personal
project to practice data collection, NLP, and basic statistics.
