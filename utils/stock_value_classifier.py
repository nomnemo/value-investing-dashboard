import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

"""# Random Forest Classifier

### Metrics:

1. P/E ratio
2. P/B ratio
3. P/S ratio
4. Sentiment score
"""

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

metrics.head()

# Drop stocks with at least one N/A for any of the ratios
metrics.dropna(inplace=True)
metrics = metrics.reset_index(drop=True)
len(metrics)

# Assign points based on whether the ratios are high or low
metrics['Overvalued Points'] = 0
metrics['Undervalued Points'] = 0

# P/E ratio gets 3 points (most important), P/B: 2 points, P/S: 1 point
for i in range(len(metrics)):
  if metrics.loc[i, 'P/E Ratio'] > 30:
    metrics.loc[i, 'Overvalued Points'] += 3
  if metrics.loc[i, 'P/B Ratio'] > 2:
    metrics.loc[i, 'Overvalued Points'] += 2
  if metrics.loc[i, 'P/S Ratio'] > 3:
    metrics.loc[i, 'Overvalued Points'] += 1

  if metrics.loc[i, 'P/E Ratio'] < 20:
    metrics.loc[i, 'Undervalued Points'] += 3
  if metrics.loc[i, 'P/B Ratio'] < 1:
    metrics.loc[i, 'Undervalued Points'] += 2
  if metrics.loc[i, 'P/S Ratio'] < 1:
    metrics.loc[i, 'Undervalued Points'] += 1

metrics.head()

# Classify stocks as overvalued or undervalued based on their points
for i in range(len(metrics)):
  if metrics.loc[i, 'Overvalued Points'] >= 3:
    metrics.loc[i, "Value"] = "Overvalued"
  elif metrics.loc[i, 'Undervalued Points'] >= 3:
    metrics.loc[i, "Value"] = "Undervalued"
  else:
    metrics.loc[i, "Value"] = "Neutral"

# Get the X and y datasets
X = metrics.drop(['Ticker', 'Overvalued Points', 'Undervalued Points', 'Value'], axis=1)
y = metrics['Value']

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

# Get a confusion matrix
print("Confusion Matrix:\n")
confusion_matrix_norm = confusion_matrix(y_test, predictions, normalize='true')
confusion_matrix_df = pd.DataFrame(confusion_matrix_norm,
                                   columns=['Neutral', 'Overvalued', 'Undervalued'],
                                   index=['Neutral', 'Overvalued', 'Undervalued'])
sns.heatmap(confusion_matrix_df, annot=True, fmt=".2f", cmap="Reds")
plt.show()

# Get the classification report
print("Classification Report:\n")
print(classification_report(y_test, predictions))

