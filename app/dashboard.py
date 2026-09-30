"""K-AI Hardware Tracker dashboard (Streamlit + Plotly).

Two tabs:
  1. Price vs Sentiment: price line + FinBERT bars on the same chart
  2. Event study: shows if sentiment spikes are followed by unusual returns

Two Y axes are used because price and sentiment ([-1, +1]) are on very
different scales.

st.cache_data is used because Streamlit re-runs the whole script every
time you touch a slider or dropdown. Without caching, that would mean
re-reading SQLite on every click. ttl=300 means data can be up to 5
minutes old, which is fine for a tool that updates once a day.
"""

import sys
from pathlib import Path

# Make the src package importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                            # noqa: E402
import plotly.graph_objects as go                               # noqa: E402
import streamlit as st                                          # noqa: E402
from plotly.subplots import make_subplots                       # noqa: E402

from src.analysis.event_study import run as run_study           # noqa: E402
from src.config import BENCHMARKS, NEWS_QUERIES, TICKERS, TICKER_TOPICS  # noqa: E402
from src.storage.db import get_connection, daily_sentiment      # noqa: E402

st.set_page_config(
    page_title="K-AI Hardware Tracker",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ K-AI Hardware Supply Chain Monitor")
st.caption(
    "Media sentiment (FinBERT) plotted against stock prices — "
    "a monitoring tool, not a prediction model."
)


# Data loading, all cached
@st.cache_data(ttl=300)
def load_prices(ticker: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT date, close, volume FROM prices WHERE ticker = ? ORDER BY date",
            conn, params=(ticker,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_sentiment(query_tags: tuple[str, ...], ticker: str) -> pd.DataFrame:
    # days of the ticker's exchange (news after the close -> next day)
    return daily_sentiment(list(query_tags), ticker=ticker)


@st.cache_data(ttl=300)
def load_recent_news(limit: int = 50) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """SELECT query_tag, title, source, published_utc,
                      sentiment_score, sentiment_label
               FROM news
               WHERE sentiment_score IS NOT NULL
               ORDER BY published_utc DESC LIMIT ?""",
            conn, params=(limit,),
        )


@st.cache_data(ttl=300)
def load_stats() -> dict:
    with get_connection() as conn:
        n_prices  = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        n_news    = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        n_scored  = conn.execute(
            "SELECT COUNT(*) FROM news WHERE sentiment_score IS NOT NULL"
        ).fetchone()[0]
        date_range = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
    return {"n_prices": n_prices, "n_news": n_news, "n_scored": n_scored,
            "date_min": date_range[0], "date_max": date_range[1]}


@st.cache_data(ttl=300)
def load_event_study(ticker: str, query_tags: tuple[str, ...], threshold: float,
                     model: str) -> dict:
    """Run the event study and return the results."""
    return run_study(ticker=ticker, query_tags=list(query_tags),
                     threshold=threshold, model=model)


# Sidebar
with st.sidebar:
    st.header("Settings")

    ticker = st.selectbox(
        "Ticker",
        options=list(TICKERS.keys()),
        format_func=lambda t: f"{TICKERS[t]} ({t})",
    )

    window_days = st.slider("Window (days)", 30, 730, 180, 30)

    # None = the topics of this ticker (TICKER_TOPICS in config).
    # I use tuples and not lists because they are passed to the cached
    # functions below.
    topic_options = {
        "Topics linked to this ticker": None,
        "All topics": tuple(NEWS_QUERIES.keys()),
    } | {v: (k,) for k, v in NEWS_QUERIES.items()}
    topic_label = st.selectbox("News topics", list(topic_options.keys()))
    selected = topic_options[topic_label]
    query_tags = selected if selected is not None else tuple(TICKER_TOPICS[ticker])
    st.caption("Topics used: " + ", ".join(query_tags))

    st.divider()
    st.subheader("Event study")
    threshold = st.slider(
        "Sentiment threshold",
        min_value=0.10, max_value=0.60, value=0.25, step=0.05,
        help="An event is a day where |avg score| exceeds this value. "
             "Lower threshold means more events.",
    )
    model_label = st.radio(
        "Normal return model",
        ["Market model", "Constant mean"],
        help="Market model: the stock is compared to its market index "
             f"({BENCHMARKS[ticker]} for this ticker). Constant mean: the "
             "average return of the stock, simpler but biased when the "
             "stock has a trend.",
    )
    model = "market" if model_label == "Market model" else "constant"

    st.divider()
    st.markdown(
        "Data: yfinance + Google News RSS  \n"
        "NLP: FinBERT (ProsusAI)  \n"
        "Score in [-1, +1] = P(positive) - P(negative)"
    )


# Top metrics
stats = load_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price rows", f"{stats['n_prices']:,}")
c2.metric("Articles collected", f"{stats['n_news']:,}")
c3.metric("Articles scored", f"{stats['n_scored']:,}")
c4.metric("Price range", f"{stats['date_min'] or '-'} to {stats['date_max'] or '-'}")

st.divider()

tab_chart, tab_study = st.tabs(["Price vs Sentiment", "Event study"])


# Tab 1: Price vs Sentiment
with tab_chart:
    prices = load_prices(ticker)
    sentiment = load_sentiment(query_tags, ticker)

    if prices.empty:
        st.warning(f"No price data for {ticker}.")
        st.stop()

    cutoff = prices["date"].max() - pd.Timedelta(days=window_days)
    prices_w = prices[prices["date"] >= cutoff]

    if not sentiment.empty:
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        sentiment_w = sentiment[sentiment["date"] >= cutoff]
    else:
        sentiment_w = pd.DataFrame(columns=["date", "avg_score", "n_articles"])

    bar_colors = (
        sentiment_w["avg_score"]
        .apply(lambda s: "rgba(52,211,153,0.75)" if s >= 0 else "rgba(248,113,113,0.75)")
        .tolist()
        if not sentiment_w.empty else []
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=prices_w["date"], y=prices_w["close"],
            name=f"Price ({ticker})",
            line=dict(color="#60a5fa", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if not sentiment_w.empty:
        fig.add_trace(
            go.Bar(
                x=sentiment_w["date"], y=sentiment_w["avg_score"],
                name="FinBERT sentiment",
                marker_color=bar_colors,
                opacity=0.75,
                customdata=sentiment_w["n_articles"],
                hovertemplate=("%{x|%Y-%m-%d}<br>Score: %{y:.3f}"
                               "<br>Articles: %{customdata}<extra></extra>"),
            ),
            secondary_y=True,
        )
        fig.add_hline(y=0, secondary_y=True, line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        hovermode="x unified", height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_yaxes(title_text="Close price", secondary_y=False, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Sentiment [-1, +1]", secondary_y=True,
                     range=[-1, 1], gridcolor="#1f2937", zeroline=False)
    fig.update_xaxes(gridcolor="#1f2937")

    st.subheader(f"{TICKERS[ticker]} - price vs sentiment ({window_days}d)")
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Recent articles (scored by FinBERT)")
    news_df = load_recent_news()
    if not news_df.empty:
        _badge = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        news_df[""] = news_df["sentiment_label"].map(_badge).fillna("⚪")
        news_df["Score"] = news_df["sentiment_score"].map("{:+.3f}".format)
        news_df["Date"] = pd.to_datetime(news_df["published_utc"]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            news_df[["", "Date", "query_tag", "source", "Score", "title"]].rename(
                columns={"query_tag": "Topic", "source": "Source", "title": "Title"}
            ),
            width="stretch", hide_index=True,
        )


# Tab 2: Event study
with tab_study:
    st.subheader(f"Event study - {TICKERS[ticker]} ({ticker})")

    with st.expander("How this event study works", expanded=False):
        st.markdown("""
**Goal**: check if a media sentiment spike is followed by an *unusual*
stock return in the days after.

**Steps:**
1. **Events** = days where the average FinBERT sentiment crosses the
   threshold. A 5-day cooldown avoids counting the same news story twice.
2. **Estimation window** [-120, -21]: this is used to find the "normal"
   return, over the ~100 trading days before the event.
   - *Market model* (default): a linear regression
     `stock return = alpha + beta * index return`. The normal return of
     a day is then alpha + beta × the index return of that day, so a day
     where the whole market goes up is not counted as abnormal.
   - *Constant mean*: just the average daily return of the stock. It is
     biased when the stock has a trend, I keep it to compare.
3. **Event window** [-5, +5]: *Abnormal Return* (AR) = actual return
   minus normal return. *CAR* = sum of AR over the window.
4. **CAAR** = average CAR across all events.
   **t-test**: checks if CAAR = 0 or not. If p > 0.05, we cannot say
   the signal is real — it could just be noise.
5. **Before vs after**: the CAR is split in days [-5, -1] and
   [0, +5]. Only the "after" part tells if returns move *after* the
   news. A big "before" part means the headlines talk about a price
   move that already happened.

**Expected result**: most likely p > 0.05. That is an honest result, it
just means this signal is not strong enough to prove anything on its own.
        """)

    with st.spinner("Computing..."):
        results = load_event_study(ticker, query_tags, threshold, model)

    if not results or results.get("all") is None:
        st.warning(
            "Not enough events for a test (minimum 5). Try lowering the "
            "sentiment threshold in the sidebar, or run the ingestion and "
            "scoring scripts again."
        )
        st.stop()

    res_all = results.get("all")

    m1, m2, m3, m4 = st.columns(4)

    if res_all:
        # main numbers = "after" window [0, +5], because this is the real
        # question of the study (do returns move AFTER the news?)
        m1.metric("Events (total)", res_all.n_events)
        m2.metric("CAAR after [0, +5]", f"{res_all.caar_post:+.2%}",
                  help="Average abnormal return from the event day to day +5")
        m3.metric("t-stat (after)", f"{res_all.t_post:+.2f}")
        sig = "Significant (p<0.05)" if res_all.p_post < 0.05 else "Not significant"
        m4.metric("p-value (after)", f"{res_all.p_post:.3f}", delta=sig,
                  delta_color="normal" if res_all.p_post < 0.05 else "off")
        st.caption(
            f"Before the event [-5, -1]: CAAR {res_all.caar_pre:+.2%} "
            f"(p={res_all.p_pre:.3f}).  "
            f"Full window [-5, +5]: CAAR {res_all.caar:+.2%} "
            f"(p={res_all.p_value:.3f})."
        )

    st.divider()

    st.markdown("**Cumulative CAAR over the event window [-5, +5]**")

    fig_car = go.Figure()

    colors = {
        "all":      ("#94a3b8", "All"),
        "positive": ("#34d399", "Positive sentiment"),
        "negative": ("#f87171", "Negative sentiment"),
    }

    for key, (color, label) in colors.items():
        res = results.get(key)
        if res is None or res.n_events < 2:
            continue
        fig_car.add_trace(go.Scatter(
            x=res.relative_days,
            y=[v * 100 for v in res.caar_by_day],   # as %
            name=f"{label} (n={res.n_events})",
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate="Day %{x}<br>CAAR: %{y:.3f}%<extra></extra>",
        ))

    fig_car.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4,
                      annotation_text="Event", annotation_position="top right")
    fig_car.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)

    fig_car.update_layout(
        template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        hovermode="x unified", height=400,
        yaxis_title="CAAR (%)",
        xaxis_title="Days relative to the event (day 0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_car.update_xaxes(gridcolor="#1f2937", dtick=1)
    fig_car.update_yaxes(gridcolor="#1f2937")

    st.plotly_chart(fig_car, width="stretch")

    st.caption(
        "Reading it: if the line rises after day 0, positive sentiment spikes "
        "are followed by outperformance. A flat line means no signal."
    )

    st.divider()
    st.markdown("**Distribution of individual CAR values**")

    if res_all and not res_all.events_df.empty:
        fig_hist = go.Figure()
        for key, (color, label) in colors.items():
            res = results.get(key)
            if res is None or res.n_events < 2:
                continue
            fig_hist.add_trace(go.Histogram(
                x=[v * 100 for v in res.events_df["car"]],
                name=label,
                marker_color=color,
                opacity=0.65,
                nbinsx=15,
                hovertemplate="CAR: %{x:.2f}%<br>Count: %{y}<extra></extra>",
            ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_hist.update_layout(
            template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            barmode="overlay", height=300,
            xaxis_title="CAR (%)",
            yaxis_title="Number of events",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_hist.update_xaxes(gridcolor="#1f2937")
        fig_hist.update_yaxes(gridcolor="#1f2937")
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    st.markdown("**Detected events**")

    if res_all and not res_all.events_df.empty:
        ev = res_all.events_df[["event_date", "direction", "avg_score", "beta",
                                "normal_return", "car_pre", "car_post", "car"]].copy()
        if model != "market":
            ev = ev.drop(columns=["beta"])   # beta only exists in the market model
        ev["event_date"] = ev["event_date"].dt.strftime("%Y-%m-%d")
        ev["direction"] = ev["direction"].map(
            {"positive": "🟢 positive", "negative": "🔴 negative"}
        )
        ev = ev.rename(columns={
            "event_date": "Date", "direction": "Direction",
            "avg_score": "Sentiment score", "beta": "Beta",
            "normal_return": "Normal return",
            "car_pre": "CAR [-5,-1]", "car_post": "CAR [0,+5]",
            "car": "CAR [-5,+5]",
        })
        st.dataframe(
            ev,
            width="stretch",
            hide_index=True,
            column_config={
                "Sentiment score":  st.column_config.NumberColumn(format="%+.3f"),
                "Beta":          st.column_config.NumberColumn(format="%.2f"),
                "Normal return": st.column_config.NumberColumn(format="%+.4f"),
                "CAR [-5,-1]":      st.column_config.NumberColumn(format="%+.4f"),
                "CAR [0,+5]":       st.column_config.NumberColumn(format="%+.4f"),
                "CAR [-5,+5]":      st.column_config.NumberColumn(format="%+.4f"),
            },
        )
