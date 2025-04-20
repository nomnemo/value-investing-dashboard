import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st


"""# Random Forest Classifier for Stock Value Classification

### Metrics:

1. P/E ratio
2. P/B ratio
3. P/S ratio
4. Sentiment score
"""

# This function creates a DataFrame with the P/E, P/B, and P/S ratios for all S&P 500 stocks
@st.cache_data(ttl=86400)  # cache for 24 hours
def get_sp500_metrics():
  # Extract all S&P500 stocks
  metrics = pd.DataFrame()
  sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
  metrics['Ticker'] = sp500

  # Assign the P/E, P/S, and P/B ratios from Yahoo Finance data
  for i in range(500):
    ticker_info = yf.Ticker(metrics['Ticker'][i])
    if 'trailingPE' in ticker_info.info:
      metrics.loc[i, 'P/E Ratio'] = ticker_info.info['trailingPE']

    if 'priceToBook' in ticker_info.info:
      metrics.loc[i, 'P/B Ratio'] = ticker_info.info['priceToBook']

    if 'priceToSalesTrailing12Months' in ticker_info.info:
      metrics.loc[i, 'P/S Ratio'] = ticker_info.info['priceToSalesTrailing12Months']

  # Drop stocks with at least one N/A for any of the ratios
  metrics.dropna(inplace=True)
  metrics = metrics.reset_index(drop=True)
  return metrics

def assign_points(df):
  # Assign points based on whether the ratios are high or low
  df['Overvalued Points'] = 0
  df['Undervalued Points'] = 0

  # P/E ratio gets 3 points (most important), P/B: 2 points, P/S: 1 point
  for i in range(len(df)):
    if df.loc[i, 'P/E Ratio'] > 30:
      df.loc[i, 'Overvalued Points'] += 3
    if df.loc[i, 'P/B Ratio'] > 2:
      df.loc[i, 'Overvalued Points'] += 2
    if df.loc[i, 'P/S Ratio'] > 3:
      df.loc[i, 'Overvalued Points'] += 1

    if df.loc[i, 'P/E Ratio'] < 20:
      df.loc[i, 'Undervalued Points'] += 3
    if df.loc[i, 'P/B Ratio'] < 1:
      df.loc[i, 'Undervalued Points'] += 2
    if df.loc[i, 'P/S Ratio'] < 1:
      df.loc[i, 'Undervalued Points'] += 1

  return df

def classify_stocks(df):
  # Classify stocks as overvalued or undervalued based on their points
  for i in range(len(df)):
    if df.loc[i, 'Overvalued Points'] >= 3:
      df.loc[i, "Value"] = "Overvalued"
    elif df.loc[i, 'Undervalued Points'] >= 3:
      df.loc[i, "Value"] = "Undervalued"
    else:
      df.loc[i, "Value"] = "Neutral"

  return df

## This function was useless
def train_classifier(X, y):
  # Split the dataset into training and testing
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Normalize the data
  scaled = StandardScaler()
  X_train = scaled.fit_transform(X_train)
  X_test = scaled.transform(X_test)

  # Run the random forest classifier and get predictions
  random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
  random_forest.fit(X_train, y_train)
  predictions = random_forest.predict(X_test)

  # Get the accuracy
  accuracy = accuracy_score(y_test, predictions)
  print("Accuracy:", accuracy)

  return y_test, predictions

## useless
def create_confusion_matrix(y_test, predictions):
  # Get a confusion matrix
  print("Confusion Matrix:\n")
  confusion_matrix_norm = confusion_matrix(y_test, predictions, normalize='true')
  confusion_matrix_df = pd.DataFrame(confusion_matrix_norm,
                                    columns=['Neutral', 'Overvalued', 'Undervalued'],
                                    index=['Neutral', 'Overvalued', 'Undervalued'])
  sns.heatmap(confusion_matrix_df, annot=True, fmt=".2f", cmap="Reds")
  plt.show()

##############################
## API functions
#################################

## This function gets the trained model and scaler
## and the S&P 500 metrics DataFrame
def get_trained_model():
    sp500_metrics = get_sp500_metrics()
    sp500_metrics = assign_points(sp500_metrics)
    sp500_metrics = classify_stocks(sp500_metrics)
    
    X = sp500_metrics.drop(['Ticker', 'Overvalued Points', 'Undervalued Points', 'Value'], axis=1)
    y = sp500_metrics['Value']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf.fit(X_scaled, y)

    return clf, scaler, sp500_metrics
  
## Given a ticker, this function classifies it as overvalued, undervalued, or neutral
## and returns the P/E, P/B, and P/S ratios
def classify_ticker(ticker, clf, scaler):
  info = yf.Ticker(ticker).info

  try:
      pe = info['trailingPE']
      pb = info['priceToBook']
      ps = info['priceToSalesTrailing12Months']
  except:
      return None, "Missing ratio(s)"

  # Format like training input
  features = [[pe, pb, ps]]
  features_scaled = scaler.transform(features)
  prediction = clf.predict(features_scaled)[0]

  return prediction, {
      "P/E": round(pe, 2),
      "P/B": round(pb, 2),
      "P/S": round(ps, 2)
  }