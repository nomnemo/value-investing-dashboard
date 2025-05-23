from utils.sentiment_analyzer import run_sentiment_analysis
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
    # Initialize the columns
    df['Overvalued Points'] = 0
    df['Undervalued Points'] = 0

    # Safely loop over rows using iterrows
    for idx, row in df.iterrows():
        ov = 0
        uv = 0

        # Overvalued points
        if row['P/E Ratio'] > 30:
            ov += 3
        if row['P/B Ratio'] > 2:
            ov += 2
        if row['P/S Ratio'] > 3:
            ov += 1

        # Undervalued points
        if row['P/E Ratio'] < 20:
            uv += 3
        if row['P/B Ratio'] < 1:
            uv += 2
        if row['P/S Ratio'] < 1:
            uv += 1

        # Assign the computed values
        df.at[idx, 'Overvalued Points'] = ov
        df.at[idx, 'Undervalued Points'] = uv

    return df

def classify_stocks(df):
    # Initialize the 'Value' column
    df['Value'] = "Neutral"  # default classification

    # Loop over rows using iterrows to safely access by index
    for idx, row in df.iterrows():
        if row['Overvalued Points'] >= 3:
            df.at[idx, "Value"] = "Overvalued"
        elif row['Undervalued Points'] >= 3:
            df.at[idx, "Value"] = "Undervalued"
        # else stays as Neutral

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

## This function returns the trained model and scaler
## and the S&P 500 metrics DataFrame
def get_trained_model():
    # Get labeled data
    metrics = get_sp500_metrics()
    metrics = assign_points(metrics)
    metrics = classify_stocks(metrics)

    # Step 1: Run sentiment analysis for tickers in the cleaned metrics
    ticker_subset = metrics['Ticker'].tolist()
    avg_sentiment = run_sentiment_analysis(ticker_subset)

    # Step 2: Merge sentiment scores into metrics
    metrics = metrics.merge(avg_sentiment, how='left', left_on='Ticker', right_index=True)
    metrics.dropna(subset=['sentiment values'], inplace=True)

    # Step 3: Extract features and labels
    X = metrics[['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'sentiment values']]
    y = metrics['Value']

    # Step 4: Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)

    return clf, scaler, metrics
  
## Given a ticker, this function classifies it as overvalued, undervalued, or neutral
## and returns the P/E, P/B, and P/S ratios
def classify_ratios(pe, pb, ps, sentiment, clf, scaler):
    if any(v is None for v in [pe, pb, ps, sentiment]):
        return None, "Missing input value(s)"

    features = [[pe, pb, ps, sentiment]]
    features_scaled = scaler.transform(features)
    prediction = clf.predict(features_scaled)[0]

    return prediction, {
        "P/E": round(pe, 2),
        "P/B": round(pb, 2),
        "P/S": round(ps, 2),
        "Sentiment": round(sentiment, 3)
    }