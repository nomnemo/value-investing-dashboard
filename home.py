import streamlit as st
st.set_page_config(page_title="Home", layout="wide")

import pandas as pd
from utils.stock_value_classifier import get_trained_model, classify_ticker

# cache the model loading to avoid reloading it every time
@st.cache_resource
def load_model():
    return get_trained_model()

clf, scaler, sp500_metrics = load_model()


st.title("Value Investing Dashboard")

# set up the layout
col1, col2 = st.columns([1, 1])

# COLUMN 1: S&P 500 Stocks Snapshot
# TODO: replace samhita's code to get the S&P 500 stocks list, maybe get it paginated?
with col1:
    st.subheader("S&P 500 Stocks Snapshot price of today ?")
    st.dataframe(sp500_metrics[['Ticker', 'P/E Ratio', 'P/B Ratio', 'P/S Ratio']])

#COLUMN 2: User Input to classify a stock ticker as overvalued, undervalued, or neutral
with col2:

    ## TITLE
    st.subheader("Analyze a Stock Ticker")

    ## TEXT INPUT
    ticker_input = st.text_input("Enter a stock ticker (e.g., AAPL):")

    ## BUTTON TO CLASSIFY
    if st.button("Classify"):
        
        ## OUTPUT CONTAINER
        st.subheader("Classification Result")
        
        ### INVLALID TICKER HANDLING:
        
        ## If the ticker is empty, show an error message and stop the execution
        if ticker_input == "": 
            st.error("Please enter a ticker symbol.")
            st.stop()
            
        ## If the ticker is invalid, i.e, not in the S&P 500 list, show an error message
        if ticker_input.upper() not in sp500_metrics['Ticker'].values:
            st.error("Ticker not found in S&P 500. Please enter a valid ticker.")
            st.stop()
        
        ### VALID TICKER HANDLING:
        label, result = classify_ticker(ticker_input.upper(), clf, scaler)
        
        ## Display the classification result
        if label == "Undervalued": ## green
            st.success(f"**{ticker_input.upper()} is classified as {label}**")
        elif label == "Overvalued": ## red
            st.warning(f"**{ticker_input.upper()} is classified as {label}**")
        elif label == "Neutral": ## neutral
            st.info(f"**{ticker_input.upper()} is classified as {label}**")
        else: ##
            st.error("Could not classify — missing data for this ticker.")

        ## Display the P/E, P/B, and P/S ratios 
        st.subheader("📊 Fundamental Ratios")

        tabs = st.tabs(["📈 P/E Ratio", "📘 P/B Ratio", "💵 P/S Ratio"])

        # --- P/E Ratio Tab ---
        with tabs[0]:
            st.metric("P/E Ratio", result.get("P/E", "N/A"))
            st.markdown("#### 📌 Formula")
            st.markdown("**P/E Ratio = Market Price per Share / Earnings per Share (EPS)**")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- A **high P/E** may indicate overvaluation or strong growth expectations.\n"
                "- A **low P/E** may indicate undervaluation or financial concerns."
            )

        # --- P/B Ratio Tab ---
        with tabs[1]:
            st.metric("P/B Ratio", result.get("P/B", "N/A"))
            st.markdown("#### 📌 Formula")
            st.markdown("**P/B Ratio = Market Price per Share / Book Value per Share**")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- A **high P/B** may suggest overvaluation or investor optimism.\n"
                "- A **low P/B** could suggest undervaluation or distress."
            )

        # --- P/S Ratio Tab ---
        with tabs[2]:
            st.metric("P/S Ratio", result.get("P/S", "N/A"))
            st.markdown("#### 📌 Formula")
            st.markdown("**P/S Ratio = Market Price per Share / Revenue per Share**")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- A **high P/S** ratio may mean the stock is expensive relative to sales.\n"
                "- A **low P/S** ratio may indicate undervaluation."
            )