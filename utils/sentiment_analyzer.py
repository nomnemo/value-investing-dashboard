from bs4 import BeautifulSoup
import requests
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_finbert():
    with st.spinner("🧠 Summoning FinBERT from the cloud..."):
      model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
      tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
      return model, tokenizer

finbert, tokenizer = load_finbert()

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"]

def get_news_headlines(ticker_data):
  article_headlines = []
  for ticker in ticker_data:
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, 'html.parser')

    headline_titles = []

    # Yahoo News headlines are typically in <h3> tags
    headlines = soup.find_all('h3')

    for headline in headlines:
        title = headline.get_text().strip()
        if title:  # Skip empty ones
            headline_titles.append(title)

    # Print results
    for headline in headline_titles:
        if headline == "News" or headline == "Life" or headline == "Entertainment" or headline == "Finance" or headline == "Sports" or headline == "New on Yahoo":
          continue
        row_data = {'ticker': ticker, 'headline': headline}
        article_headlines.append(row_data)
  return pd.DataFrame(article_headlines)

def calculate_sentiment_values(headlines_data):
  sentiment_values = []

  for headline in headlines_data['headline']:
      inputs = tokenizer(headline, return_tensors="pt", padding=True)
      # get positive sentiment value. greater value = more positive sentiment, lesser value = less positive sentiment
      sentiment_value = finbert(**inputs)[0].detach().numpy()[0][1]

      sentiment_values.append(sentiment_value)

  headlines_data['sentiment values'] = sentiment_values
  return headlines_data

# get a average sentiment score for a list of tickers
def get_avg_sentiment_value(sentiment_df):
  averaged_sentiment = sentiment_df.groupby('ticker')['sentiment values'].mean()
  return averaged_sentiment

# tickers is a dataframe containing a column called 'tickers', that has all the tickers
def run_sentiment_analysis(tickers):
  # scrapes headline data based on inputted ticker
  sp_headlines = get_news_headlines(tickers)
  # adds a column containing the sentiment value for each headline
  sp_headlines_sentiment = calculate_sentiment_values(sp_headlines)
  # groups sentiment values by ticker, then averages the sentiment value
  averaged_sentiment = get_avg_sentiment_value(sp_headlines_sentiment)

  return averaged_sentiment

## given a ticker, scrapes the news headlines from Yahoo Finance
def get_news_headlines_for_ticker(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, 'html.parser')

    # Extract <h3> headlines
    headlines = [h.get_text().strip() for h in soup.find_all("h3")]
    
    # Filter out irrelevant categories
    skip = {"News", "Life", "Entertainment", "Finance", "Sports", "New on Yahoo"}
    headlines = [h for h in headlines if h not in skip]

    return headlines

# given a list of headlines, returns the sentiment score for each headline
def get_sentiment_score_from_headlines(headlines):
    if not headlines:
        return 0.0

    sentiment_scores = []
    for h in headlines:
        inputs = tokenizer(h, return_tensors="pt", padding=True)
        output = finbert(**inputs)[0].detach().numpy()[0]
        sentiment_scores.append(output[1])  # logit for positive class

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return round(avg_sentiment, 3)

# given a ticker, returns the sentiment score for the ticker, along with the headlines of the news articles
# used to get the sentiment score for the ticker
def analyze_sentiment_for_ticker(ticker):
  headlines = get_news_headlines_for_ticker(ticker)
  sentiment = get_sentiment_score_from_headlines(headlines)
  return sentiment, headlines
