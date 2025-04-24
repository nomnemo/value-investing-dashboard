import streamlit as st
st.set_page_config(page_title="Home", layout="wide")

import pandas as pd
import yfinance as yf # type: ignore

from utils.stock_value_classifier import classify_ratios, get_trained_model
from utils.sentiment_analyzer import analyze_sentiment_for_ticker

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
st.title("Value Investing Dashboard ")

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
        
        ticker_upper = ticker_input.upper()
        label = None
        result = None
        
        pe = None
        ps = None
        pb = None
        sentiment_score = 0
        headlines = []
            
        if  ticker_upper in sp500_metrics['Ticker'].values:
            pe = sp500_metrics.loc[sp500_metrics['Ticker'] == ticker_upper, 'P/E Ratio'].values[0]
            pb = sp500_metrics.loc[sp500_metrics['Ticker'] == ticker_upper, 'P/B Ratio'].values[0]
            ps = sp500_metrics.loc[sp500_metrics['Ticker'] == ticker_upper, 'P/S Ratio'].values[0]
        else: 
            # try to pull the pe, ps, and pb ratios from the ticker from Yahoo Finance
            # if it fails, show an error message
            
            try:
                info = yf.Ticker(ticker_upper).info
                pe = info.get('trailingPE')
                pb = info.get('priceToBook')
                ps = info.get('priceToSalesTrailing12Months')
            except Exception as e:
                st.error("Failed to fetch data for this ticker. Please enter a valid stock symbol.")
                st.stop()

        if None in [pe, pb, ps]:
            st.error("Could not retrieve all ratios from Yahoo Finance. Please check the ticker symbol and try again.")
            st.stop()
        
        ## Get the sentiment score and headlines for the ticker
        sentiment_score, headlines = analyze_sentiment_for_ticker(ticker_upper)
        label, result = classify_ratios(pe, pb, ps, sentiment_score, clf, scaler)
        
        
        ## Display the classification result
        if label == "Undervalued": ## green
            st.success(f"**{ticker_upper} is classified as {label}**")
        elif label == "Overvalued": ## red
            st.warning(f"**{ticker_upper} is classified as {label}**")
        elif label == "Neutral": ## neutral
            st.info(f"**{ticker_upper} is classified as {label}**")
        else: ##
            st.error("Could not classify — missing data for this ticker.")

        ## Display the P/E, P/B, and P/S ratios 
        st.subheader("📊 Fundamental Ratios")

        tabs = st.tabs(["📈 P/E Ratio", "📘 P/B Ratio", "💵 P/S Ratio", "📰 Sentiment"])

        # --- P/E Ratio Tab ---
        with tabs[0]:
            st.metric("P/E Ratio", result.get("P/E", "N/A"))
            st.subheader("📌 Formula")
            st.markdown(r"- The P/E ratio tells you how much investors are willing to pay today for \$1 of earnings.")
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
            st.metric("🧠 Sentiment Score", sentiment_score)

            st.markdown("#### 🗞 News Headlines Used for Sentiment")
            st.dataframe(pd.DataFrame({'Headline': headlines}))



import streamlit as st
from utils.stock_value_classifier import assign_points, classify_stocks

st.divider()
st.subheader("🏆 Top 5 Stock Picks")

if st.button("Generate Rankings"):
    with st.spinner("Fetching stock data..."):
        df = assign_points(sp500_metrics.copy())
        df = classify_stocks(df)

        # Compute category-based percentiles
        def assign_category_percentile(df, group_col, score_col, new_col):
            df[new_col] = None
            for label in df[group_col].unique():
                group = df[df[group_col] == label]
                df.loc[group.index, new_col] = group[score_col].rank(pct=True, method="first")
            return df

        df = assign_category_percentile(df, "Value", "Undervalued Points", "Undervalued Percentile")
        df = assign_category_percentile(df, "Value", "Overvalued Points", "Overvalued Percentile")

        # Filter and display
        top_undervalued = df[df["Value"] == "Undervalued"].sort_values(by="Undervalued Points", ascending=False).head(5)
        top_overvalued = df[df["Value"] == "Overvalued"].sort_values(by="Overvalued Points", ascending=False).head(5)

        st.markdown("Top 5 Undervalued Stocks")
        st.dataframe(
            top_undervalued.drop(columns=["Overvalued Points", "Undervalued Points", "Value", "Overvalued Percentile"])
            .set_index("Ticker"),
            use_container_width=True
        )

        st.caption("These stocks are currently undervalued")

        st.markdown("Top 5 Overvalued Stocks")
        st.dataframe(
            top_overvalued.drop(columns=["Overvalued Points", "Undervalued Points", "Value", "Undervalued Percentile"])
            .set_index("Ticker"),
            use_container_width=True
        )

        st.caption("These stocks are currently overvalued")

        with st.expander("What does the Percentile Score mean?"):
            st.markdown(
                "The **percentile score** shows how a stock ranks among others in the same category:\n\n"
                " Example: A score of **0.98** → top 2% of stocks in that group."
            )
