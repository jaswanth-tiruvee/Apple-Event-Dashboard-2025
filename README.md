# Apple-Event-Dashboard-2025



# Apple Event Media Intelligence Dashboard

An interactive Streamlit dashboard for analyzing media coverage of Apple’s **iPhone 17 launch**.  
The app ingests live Google News RSS feeds, performs sentiment analysis using transformers (RoBERTa) or VADER, applies semantic topic modeling with BERTopic, and visualizes insights in an interactive, exportable interface.

---

## Features

- **Live data ingestion** – fetches recent articles from Google News RSS for configurable keywords (default: iPhone 17, iPhone Air, Apple Event).  
- **Sentiment analysis** – transformer-based RoBERTa model (context-aware) with fallback to VADER.  
- **Topic modeling** – BERTopic for semantic clusters, with fallback to TF-IDF + KMeans.  
- **Interactive dashboard** – Streamlit app with multiple analysis tabs:  
  - Key metrics: total articles, sentiment percentages, buzz momentum.  
  - Sentiment distribution and word clouds.  
  - Article timeline over time.  
  - Publisher-level sentiment analysis.  
  - Topic clusters with representative headlines.  
- **Data export** – download filtered article datasets as CSV for further analysis.  

---

## Quickstart

Clone the repository and start exploring the visualizations.
