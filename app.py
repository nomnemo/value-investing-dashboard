import streamlit as st
import pandas as pd
from utils.dummy_classifier import train_classifier, classify_ticker


st.set_page_config(layout="wide")
st.title("Value Investing Dashboard")

clf, scaler, sp500_metrics = train_classifier() # assume this works

# set up the layout
col1, col2 = st.columns([1, 1])

# for the first column, we can show the S&P 500 stocks snapshot
with col1:
    st.subheader("S&P 500 Stocks Snapshot")
    st.dataframe(sp500_metrics[['Ticker', 'P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'Value']])

# for the second column, we can allow the user to input a stock ticker
# and classify it
# can handler errors if the ticker is not found or if any of the input data is missing
with col2:
    st.subheader("Analyze a Stock Ticker")
    ticker_input = st.text_input("Enter a stock ticker (e.g., AAPL):")

    if st.button("Classify"):
        label, result = classify_ticker(ticker_input.upper(), clf, scaler)
        if label is None:
            st.error("Could not classify — missing data for this ticker.")
        else:
            st.success(f"**{ticker_input.upper()} is classified as {label}**")
            st.write("Fundamental Ratios:")
            for k, v in result.items():
                st.metric(k, v)
