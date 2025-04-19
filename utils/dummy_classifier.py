import random

def train_classifier():
    # Simulate a trained classifier and scaler
    return "dummy_clf", "dummy_scaler", get_dummy_metrics()

def classify_ticker(ticker, clf, scaler):
    possible_labels = ["Overvalued", "Undervalued", "Neutral"]
    label = random.choice(possible_labels)
    scores = {
        "P/E": round(random.uniform(10, 40), 2),
        "P/B": round(random.uniform(0.5, 5), 2),
        "P/S": round(random.uniform(0.5, 5), 2)
    }
    return label, scores

def get_dummy_metrics():
    import pandas as pd
    import numpy as np

    tickers = ["AAPL", "GOOG", "MSFT", "TSLA", "META"]
    data = {
        "Ticker": tickers,
        "P/E Ratio": np.random.uniform(10, 40, size=5),
        "P/B Ratio": np.random.uniform(0.5, 5, size=5),
        "P/S Ratio": np.random.uniform(0.5, 5, size=5),
        "Value": np.random.choice(["Overvalued", "Undervalued", "Neutral"], size=5)
    }
    return pd.DataFrame(data)
