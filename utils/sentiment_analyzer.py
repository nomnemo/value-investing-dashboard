
import pandas as pd
# news api for scraping data for sentiment analysis:
import requests
import requests
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta

"""# Sentiment Analysis
Scrape data using News API
"""

api_key = 'a0191a22e5454894ade2a57228f61329'
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"]

sp500.head()

print(data['message'])


ticker = "AAPL"
url = f"https://finance.yahoo.com/quote/{ticker}/news"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

for item in soup.select('li.js-stream-content'):
    headline = item.find('h2')
    print(headline)
    if headline:
        print(headline.text.strip())


# Get list of S&P 500 tickers and company names
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(sp500_url)[0]
tickers = table['Symbol'].tolist()[0:10]
names = table['Security'].tolist()[0:10]
ticker_map = dict(zip(tickers, names))

# Time filter for the past 7 days
cutoff_date = datetime.now() - timedelta(days=7)

# Output container
articles = []

# Loop through tickers
for ticker in tickers:
    print(f"Scraping {ticker}...")
    search_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Get article links
        for a in soup.select('a[href^="/news/"]'):
            href = a['href']
            link = f"https://finance.yahoo.com{href}"
            headline = a.text.strip()

            # Visit each article link
            try:
                article_r = requests.get(link, headers=headers)
                article_soup = BeautifulSoup(article_r.text, 'html.parser')
                paragraphs = article_soup.find_all('p')
                article_text = ' '.join(p.get_text() for p in paragraphs)

                # Optional: Extract article date
                time_tag = article_soup.find('time')
                if time_tag and time_tag.has_attr('datetime'):
                    pub_date = datetime.fromisoformat(time_tag['datetime'].replace("Z", "+00:00"))
                    if pub_date < cutoff_date:
                        continue  # Skip old articles
                else:
                    pub_date = None  # If not found, include anyway

                articles.append({
                    "headline": headline,
                    "article_body": article_text,
                    "stock_ticker": ticker,
                    "company_name": ticker_map[ticker]
                })

                time.sleep(1)  # Be polite to Yahoo servers

            except Exception as e:
                print(f"Failed to scrape article: {link} ({e})")
                continue

        time.sleep(2)  # Sleep between ticker scrapes

    except Exception as e:
        print(f"Failed to scrape {ticker}: {e}")

# Create DataFrame
df = pd.DataFrame(articles)
print(df.head())

from datetime import datetime
from datetime import timedelta

def get_all_news_data():
  articles = []
  # current_date = datetime.strptime(start_date, '%Y-%m-%d')
  # end_date = datetime.strptime(end_date, '%Y-%m-%d')

  for ticker, security in sp500:
    # get news data for each day

    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={security}&mode=artlist&maxrecords={100}&format=json"
    )
    response = requests.get(url)
    article_data = response.json()['articles']
    # print(article_data.keys())
    # article = fetch_news(api_key, ticker, curr_date_str)
    for article in article_data['articles']:
      articles.append({'ticker': ticker, 'security': security, 'article': article['content']})


      # current_date += timedelta(days=1)
  return pd.DataFrame(articles)

sp500_news_data = get_all_news_data()
sp500_news_data.head()

from bs4 import BeautifulSoup as BeautifulSoup
import requests

html = requests.get('https://news.yahoo.com').text
soup = BeautifulSoup(html, 'html.parser')

headline_titles = []
headlines = soup.find_all('h3')
for headline in headlines:
  headline = str(headline)
  title = re.findall('\>[0-9a-zA-Z\s\,\.\'\"\-\:]*')
  title.sort(reverse=True)

  headline_titles.append(title[0][1:])

for headline in headline_titles:
  print(headline)

"""**USE THIS: **"""

from bs4 import BeautifulSoup
import requests

def get_news_headlines():
  article_headlines = []
  for ticker in sp500:
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    html = requests.get('https://finance.yahoo.com/quote/AAPL/news/', headers={"User-Agent": "Mozilla/5.0"}).text
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

sp_headlines = get_news_headlines()
sp_headlines.head()

sp_headlines.to_csv('sp_500_headlines.csv', index=False)
from google.colab import files
files.download('sp_500_headlines.csv')

"""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"""

import pandas as pd
from google.colab import files
import io
uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['sp_500_headlines.csv']))

df.columns

from transformers import BertTokenizer, BertForSequenceClassification

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

sentiment_values = []

for headline in df['headline']:
    inputs = tokenizer(headline, return_tensors="pt", padding=True)
    sentiment_value = finbert(**inputs)[0].detach().numpy()[0]

    # sentiment_value = labels[np.argmax(outputs.detach().numpy())]
    # print(headline, '----', sentiment_value.detach().numpy()[0])
    # print('#######################################################')
    sentiment_values.append(sentiment_value)

df['sentiment values'] = sentiment_values
df.head()

html = requests.get('https://finance.yahoo.com/quote/TSLA/news/', headers={"User-Agent": "Mozilla/5.0"}).text
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
    print(headline)