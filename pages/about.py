import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("📚 About This Project")

st.markdown("""
### Value Investing Dashboard

This project is built by [Your Team Name] for [Your Class/University] to help investors explore and analyze stocks using value investing principles.

**Key Features:**
- Analyze S&P 500 stocks using P/E, P/B, and P/S ratios
- Visualize how each stock is classified (overvalued, undervalued, neutral)
- View top 5 ranked stocks by value
- Includes sentiment analysis component (in progress)

**Tech Stack:**
- Python, Streamlit, yfinance, scikit-learn
- Machine learning via Random Forest
- Sentiment analysis planned via FinBERT

**Team Members:**
- Nomin Ganzorig
- [Add others here]
""")
