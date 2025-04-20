import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("📘 About This Project")

st.markdown("""
### Value Investing Dashboard

This project was developed as the final project for **STAT 486: Market Models** at Rice University (Spring 2025). It explores value investing — an investment strategy focused on identifying undervalued stocks based on key financial ratios. Inspired by the hands-on research process many retail investors follow, our goal was to simplify and streamline this workflow through an interactive and intelligent dashboard.

The app allows users to analyze S&P 500 stocks using core valuation metrics like **P/E**, **P/B**, and **P/S** ratios. It also features a machine learning classifier (Random Forest) that categorizes each stock as **undervalued**, **overvalued**, or **neutral**, along with real-time **sentiment analysis** using FinBERT to provide additional context. The platform is built using **Python**, **Streamlit**, and **scikit-learn**, with data sourced from **Yahoo Finance**.

---

**🔍 Key Features**
- Analyze and classify S&P 500 stocks using financial ratios
- Visualize classification results and valuation scores
- Get real-time sentiment score and news headlines
- View top 5 most undervalued or overvalued stocks

**🛠 Tech Stack**
- Python, Streamlit, yfinance, scikit-learn
- Machine Learning: Random Forest Classifier
- NLP: FinBERT for sentiment analysis
- Web scraping via BeautifulSoup

**👥 Team Members**
Nomin, Anna, Samhita, Misha, Diya, Dasha, Ruchi
""")
