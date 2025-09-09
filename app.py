
# Apple Event Media Intelligence – Streamlit App (no keys)

# Features:
# - Live ingest from Google News RSS for your keywords
# - Transformer sentiment (RoBERTa) with auto-fallback to VADER
# - Topic modeling (BERTopic) with fallback to TF-IDF + KMeans
# - Interactive filters, KPIs, timelines, word clouds, source breakdown
# - CSV export
#

import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import feedparser
import tldextract

import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Try to import HuggingFace pipeline (advanced sentiment)
USE_TRANSFORMER = True
try:
    from transformers import pipeline
except Exception:
    USE_TRANSFORMER = False

# VADER fallback sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
vader = SentimentIntensityAnalyzer()

# Topic modeling: try BERTopic; otherwise fallback to KMeans
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    USE_BERTOPIC = True
except ImportError:
    USE_BERTOPIC = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------------------------
# Helpers
# ---------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def domain_from_url(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    except Exception:
        return "unknown"

def gnews_rss_url(query: str, lang="en", region="US"):
    # Encoded query for Google News RSS
    q = query.replace(" ", "+")
    return f"https://news.google.com/rss/search?q={q}&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"

def fetch_rss(query_list, max_items_per_feed=80, sleep_sec=0.4):
    rows = []
    for q in query_list:
        url = gnews_rss_url(q)
        fp = feedparser.parse(url)
        for e in fp.entries[:max_items_per_feed]:
            title = e.get("title","").strip()
            summary = e.get("summary","").strip()
            link = e.get("link","")
            if "published_parsed" in e and e.published_parsed:
                ts = datetime(*e.published_parsed[:6])
            else:
                ts = datetime.utcnow()
            rows.append({
                "query": q,
                "title": title,
                "summary": summary,
                "link": link,
                "published": ts
            })
        time.sleep(sleep_sec)
    df = pd.DataFrame(rows).drop_duplicates(subset=["title","link"]).reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    # Light, accurate, widely used sentiment head
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def map_hf_label(label: str) -> str:
    label = label.upper()
    if "POS" in label:
        return "Positive"
    if "NEG" in label:
        return "Negative"
    return "Neutral"

def sentiment_scores(texts):
    """Return labels & scores using transformer if available, else VADER."""
    if USE_TRANSFORMER:
        try:
            clf = load_sentiment_pipeline()
            preds = clf(texts, truncation=True, max_length=512)
            labels = [map_hf_label(p["label"]) for p in preds]
            scores = [float(p["score"]) for p in preds]
            return labels, scores, "transformer"
        except Exception:
            pass  # fallback to VADER
    # VADER
    labels, scores = [], []
    for t in texts:
        score = vader.polarity_scores(t)["compound"]
        if score >= 0.05:
            labels.append("Positive")
        elif score <= -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")
        scores.append(abs(score))
    return labels, scores, "vader"

def wordcloud_from_text(texts, title="Word Cloud"):
    text = " ".join([t for t in texts if isinstance(t, str)])
    if len(text.split()) < 3:
        st.info("Not enough text to render a word cloud.")
        return
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

def fallback_topics(texts, k=None):
    """Simple TF-IDF + KMeans topics when BERTopic isn't available."""
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    if len(texts) < 6:
        return None, None, None  # too small
    vec = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    if not k:
        k = min(6, max(2, len(texts)//10))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    # top terms
    centers = km.cluster_centers_
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for i, center in enumerate(centers):
        idx = center.argsort()[-10:][::-1]
        top_terms[i] = terms[idx].tolist()
    return labels, top_terms, k

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Apple Event Media Intelligence", layout="wide")
st.title(" Apple Event – Media Intelligence (Live RSS • No Keys)")

with st.sidebar:
    st.subheader("Settings")
    keywords = st.text_input(
        "Keywords (comma-separated)",
        value="iPhone 17, iPhone Air, Apple Event"
    )
    days_back = st.slider("Lookback (days)", 1, 14, 5)
    lang = st.selectbox("Language", ["en"], index=0)
    region = st.selectbox("Region", ["US"], index=0)
    max_items = st.slider("Max items per feed", 20, 120, 80)
    do_transformer = st.checkbox("Use Transformer Sentiment (if available)", True)
    if not do_transformer:
        USE_TRANSFORMER = False
    do_bertopic = st.checkbox("Use BERTopic (if available)", False)
    if do_bertopic and not USE_BERTOPIC:
        st.info("BERTopic not installed in this environment; fallback will be used.")

    run_btn = st.button("Ingest / Refresh")

# Session state cache
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# ---------------------------
# Data ingest
# ---------------------------
if run_btn:
    qs = [q.strip() for q in keywords.split(",") if q.strip()]
    with st.spinner("Fetching Google News RSS…"):
        df = fetch_rss(qs, max_items_per_feed=max_items, sleep_sec=0.3)
    if df.empty:
        st.error("No items found. Try different keywords or come back later.")
    else:
        # Time filter
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        df = df[df["published"] >= cutoff].copy()
        # Enrich
        df["title_clean"] = df["title"].apply(clean_text)
        df["summary_clean"] = df["summary"].apply(clean_text)
        df["text"] = (df["title_clean"] + " " + df["summary_clean"]).str.strip()
        df["domain"] = df["link"].apply(domain_from_url)
        df["hour"] = pd.to_datetime(df["published"]).dt.floor("H")

        # Sentiment
        labels, scores, engine = sentiment_scores(df["text"].tolist())
        df["sentiment"] = labels
        df["sent_score"] = scores
        df["sent_engine"] = engine

        st.session_state.df = df

# If we have data, draw the app
df = st.session_state.df
if df.empty:
    st.info("➡️ Enter keywords and click **Ingest / Refresh** to load live articles.")
    st.stop()

# Filters row
st.markdown("### Filters")
colf1, colf2, colf3, colf4 = st.columns(4)
with colf1:
    q_pick = st.multiselect("Queries", sorted(df["query"].unique()), default=list(df["query"].unique()))
with colf2:
    src_pick = st.multiselect("Sources", sorted(df["domain"].unique()), default=sorted(df["domain"].unique())[:15])
with colf3:
    sent_pick = st.multiselect("Sentiment", ["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
with colf4:
    start_date = st.date_input("From date", value=(datetime.utcnow()-timedelta(days=days_back)).date())

mask = (
    df["query"].isin(q_pick)
    & df["domain"].isin(src_pick)
    & df["sentiment"].isin(sent_pick)
    & (df["published"] >= pd.Timestamp(start_date))
)
dfv = df[mask].copy()
st.caption(f"Showing {len(dfv)} of {len(df)} articles  •  Sentiment engine: **{df['sent_engine'].iloc[0].upper()}**")

# ---------------------------
# KPIs
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Articles", len(dfv))
with col2:
    pos = (dfv["sentiment"]=="Positive").mean()
    st.metric("Positive %", f"{pos*100:,.1f}%")
with col3:
    neg = (dfv["sentiment"]=="Negative").mean()
    st.metric("Negative %", f"{neg*100:,.1f}%")
with col4:
    # very simple "buzz momentum": last 24h vs prior 24h
    now = dfv["hour"].max()
    last = dfv[dfv["hour"]>=now-timedelta(hours=24)]
    prior = dfv[(dfv["hour"]<now-timedelta(hours=24)) & (dfv["hour"]>=now-timedelta(hours=48))]
    momentum = (len(last) - max(len(prior),1)) / max(len(prior),1)
    st.metric("Buzz Momentum (24h)", f"{momentum*100:,.0f}%")

# ---------------------------
# Tabs
# ---------------------------
tab_overview, tab_sent, tab_topics, tab_sources, tab_export = st.tabs(
    ["Overview", "Sentiment", "Topics", "Sources", "Export"]
)

with tab_overview:
    st.subheader("Article Timeline")
    vol = dfv.groupby("hour").size()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(vol.index, vol.values)
    ax.set_title("Article Volume Over Time")
    ax.set_ylabel("Count")
    ax.set_xlabel("Hour (UTC)")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("Headlines (sample)")
    st.dataframe(dfv[["published","domain","query","title","sentiment","link"]].sort_values("published", ascending=False).head(25), use_container_width=True)

with tab_sent:
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(5,3))
    dfv["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"]).plot(kind="bar", ax=ax)
    ax.set_ylabel("Articles")
    st.pyplot(fig)

    st.subheader("Word Cloud (All Articles)")
    wordcloud_from_text(dfv["text"].tolist(), title="News Word Cloud")

    st.subheader("Word Clouds by Sentiment")
    for label in ["Positive","Negative"]:
        st.markdown(f"**{label}**")
        wordcloud_from_text(dfv.loc[dfv["sentiment"]==label, "text"].tolist(),
                            title=f"{label} Word Cloud")

with tab_topics:
    st.subheader("Topic Modeling")
    texts = dfv["text"].tolist()
    if len([t for t in texts if isinstance(t,str) and t.strip()]) < 6:
        st.info("Not enough articles for topic modeling. Try broadening filters.")
    else:
        if USE_BERTOPIC and st.checkbox("Use BERTopic (embedding-based)", value=False):
            with st.spinner("Fitting BERTopic…"):
                # Small, fast SBERT
                emb_model = SentenceTransformer("all-MiniLM-L6-v2")
                topic_model = BERTopic(verbose=False, embedding_model=emb_model, min_topic_size=max(5, len(texts)//20))
                topics, _ = topic_model.fit_transform(texts)
                dfv["topic"] = topics
                st.write(topic_model.get_topic_info().head(10))
                # Show top 3 topics with example headlines
                top_topics = dfv["topic"].value_counts().head(3).index.tolist()
                for t in top_topics:
                    st.markdown(f"**Topic {t} — examples**")
                    st.write(dfv[dfv["topic"]==t][["title","domain"]].head(5))
        else:
            labels, top_terms, k = fallback_topics(texts)
            if labels is None:
                st.info("Not enough documents for clustering fallback.")
            else:
                dfv["topic"] = labels
                st.markdown(f"**KMeans topics (k={k}) — top terms**")
                for c, terms in top_terms.items():
                    st.write(f"Topic {c}: ", ", ".join(terms))
                st.markdown("**Examples per topic**")
                for c in sorted(dfv["topic"].unique()):
                    sample = dfv[dfv["topic"]==c][["title","domain"]].head(5)
                    st.write(f"Topic {c}:")
                    st.dataframe(sample, use_container_width=True)

with tab_sources:
    st.subheader("Sentiment by Source (Top Domains)")
    top_domains = dfv["domain"].value_counts().head(12).index.tolist()
    sub = (dfv[dfv["domain"].isin(top_domains)]
           .groupby(["domain","sentiment"]).size().unstack(fill_value=0)
           .reindex(columns=["Positive","Neutral","Negative"]))
    fig, ax = plt.subplots(figsize=(10,4))
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("Articles")
    ax.set_title("Top Domains × Sentiment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top Domains")
st.dataframe(dfv["domain"].value_counts().reset_index().rename(columns={"index": "domain", "domain": "articles"}), use_container_width=True)

with tab_export:
    st.subheader("Download data")
    csv = dfv.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv, file_name="apple_event_media_intelligence.csv", mime="text/csv")
    st.caption("Includes: published, domain, query, title, summary, sentiment, score, topic (if computed), and links.")
