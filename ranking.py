import streamlit as st
from utils.dummy_classifier import train_classifier

st.set_page_config(page_title="Stock Rankings", layout="wide")
st.title("🏆 Top Value Stock Picks")

clf, scaler, sp500_metrics = train_classifier()

top_undervalued = sp500_metrics[sp500_metrics['Value'] == 'Undervalued'].head(5)
top_overvalued = sp500_metrics[sp500_metrics['Value'] == 'Overvalued'].head(5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 5 Undervalued Stocks")
    st.dataframe(top_undervalued)

with col2:
    st.subheader("Top 5 Overvalued Stocks")
    st.dataframe(top_overvalued)