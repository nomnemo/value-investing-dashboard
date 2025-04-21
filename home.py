import streamlit as st
st.set_page_config(page_title="Home", layout="wide")
import pandas as pd

from utils.stock_value_classifier import get_trained_model, classify_ticker
from utils.sentiment_analyzer import (
    get_news_headlines_for_ticker,
    get_sentiment_score_from_headlines
)

# 1. Caches the heavy computation
@st.cache_resource
def get_model():
    return get_trained_model()

# 2. Wraps it in a visible spinner
def load_model_with_spinner():
    with st.spinner("📡 Fetching financial data from Yahoo... please hold the line 📞"):
        return get_model()

# Load
clf, scaler, sp500_metrics = load_model_with_spinner()

## TITLE
st.title("Value Investing Dashboard")

# LAYOUT
col1, col2 = st.columns([1, 1])

# COLUMN 1: S&P 500 Stocks Snapshot
# TODO: replace samhita's code to get the S&P 500 stocks list, maybe get it paginated?
with col1:
    st.subheader("S&P 500 Stocks metrics today")
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

        tabs = st.tabs(["📈 P/E Ratio", "📘 P/B Ratio", "💵 P/S Ratio", "📰 Sentiment"])

        # --- P/E Ratio Tab ---
        with tabs[0]:
            st.metric("P/E Ratio", result.get("P/E", "N/A"))
            st.subheader("📌 Formula")
            st.latex(r"\text{P/E Ratio} = \frac{\text{Market Price per Share}}{\text{Earnings per Share (EPS)}}")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- The P/E ratio tells you how much investors are willing to pay today for \$1 of earnings.\n"
                "- A **high P/E** may indicate overvaluation or strong growth expectations.\n"
                "- A **low P/E** may indicate undervaluation or financial concerns."
            )

        # --- P/B Ratio Tab ---
        with tabs[1]:
            st.metric("P/B Ratio", result.get("P/B", "N/A"))
            st.markdown("#### 📌 Formula")
            st.latex(r"\text{P/B Ratio} = \frac{\text{Market Price per Share}}{\text{Book Value per Share}}")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- The P/B ratio compares the stock price to the company's net assets (what it's worth on paper).\n"
                "- A **high P/B** may suggest overvaluation or investor optimism.\n"
                "- A **low P/B** could suggest undervaluation or distress."
            )

        # --- P/S Ratio Tab ---
        with tabs[2]:
            st.metric("P/S Ratio", result.get("P/S", "N/A"))
            st.markdown("#### 📌 Formula")
            st.latex(r"\text{P/S Ratio} = \frac{\text{Market Price per Share}}{\text{Revenue per Share}}")
            st.markdown("#### 💡 Interpretation")
            st.markdown(
                "- The P/S ratio shows how much investors are paying for each dollar of a company’s sales.\n"
                "- A **high P/S** ratio may mean the stock is expensive relative to sales.\n"
                "- A **low P/S** ratio may indicate undervaluation."
            )
        with tabs[3]:

            # Get Yahoo Finance headlines for the ticker
            headlines = get_news_headlines_for_ticker(ticker_input.upper())

            if headlines:
                # Compute sentiment score from headlines
                sentiment_score = get_sentiment_score_from_headlines(headlines)

                st.metric("🧠 Sentiment Score", sentiment_score)

                st.markdown("#### 🗞 News Headlines Used for Sentiment")
                st.dataframe(pd.DataFrame({'Headline': headlines}))
            else:
                st.warning("No news headlines found for this ticker.")

st.divider()
st.subheader("Top 5 Stock Picks")

if st.button("Generate Rankings"):
    with st.spinner("Fetching stock data..."):
        from utils.stock_value_classifier import assign_points, classify_stocks

        df = assign_points(sp500_metrics.copy())
        df = classify_stocks(df)

        top_undervalued = df[df["Value"] == "Undervalued"].sort_values(by="Undervalued Points", ascending=False).head(5)
        top_overvalued = df[df["Value"] == "Overvalued"].sort_values(by="Overvalued Points", ascending=False).head(5)

        top_undervalued = top_undervalued.rename(columns={"Value": "Classification"})
        top_overvalued = top_overvalued.rename(columns={"Value": "Classification"})

        st.markdown("###Top 5 Undervalued Stocks")
        st.dataframe(top_undervalued.set_index("Ticker"), use_container_width=True)
        st.caption("These stocks are currently undervalued")

        st.markdown("###Top 5 Overvalued Stocks")
        st.dataframe(top_overvalued.set_index("Ticker"), use_container_width=True)
        st.caption("These stocks are currently overvalued")