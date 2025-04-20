from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification


## takes the dataframe of the S&P 500 stocks tickers
## right now we want to make it so that it takes 1 ticker at a time
## otherwise it will take a long time to scrape the data
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

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

def calculate_sentiment_values(headlines_data):
  sentiment_values = []

  for headline in headlines_data['headline']:
      inputs = tokenizer(headline, return_tensors="pt", padding=True)
      # get positive sentiment value. greater value = more positive sentiment, lesser value = less positive sentiment
      sentiment_value = finbert(**inputs)[0].detach().numpy()[0][1]

      sentiment_values.append(sentiment_value)

  headlines_data['sentiment values'] = sentiment_values
  return headlines_data

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
