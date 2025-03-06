import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
import os

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_monthly_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.resample('ME').last()['Close']

def get_stock_forecast_model(ticker, target_price, target_range=0.05, max_months=60):
    logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    # Fetch monthly stock data
    data = get_monthly_stock_data(ticker, start_date, end_date)
    if data.empty:
        logger.error("Invalid ticker or no data found!")
        return None
    logger.info(f"Successfully fetched data. Records: {len(data)}")

    model_filename = f"sarima/{ticker}.pkl"
    if not os.path.exists(model_filename):
        logger.error(f"Model file {model_filename} not found!")
        return None
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        logger.info("SARIMA model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SARIMA model: {e}")
        return None

    try:
        forecast = model.forecast(steps=max_months)  # Forecast up to max_months (60)
        logger.info("SARIMA forecast generated.")
    except Exception as e:
        logger.error(f"SARIMA forecast failed: {e}")
        return None

    future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, max_months + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})

    # Find the first month where target price is reached
    hit_index = (forecast >= target_price * (1 - target_range)) & (forecast <= target_price * (1 + target_range))
    
    if hit_index.any():
        hit_month = (hit_index.argmax() + 1)  # Index starts from 0, so add 1
        hit_month_years = round((hit_month / 12),2)  # Convert months to years
        hit_msg = f"🎯 Target price is likely to be reached in **{hit_month} months/ {hit_month_years} Years** ."
    else:
        hit_month = None
        hit_msg = "⚠️ Target price may not be reached within 5 years."

    return data, pred_df, hit_month, hit_msg

# ------------------------------
# STREAMLIT UI
# ------------------------------

st.set_page_config(page_title="Financial Consultant", page_icon="📈")

st.warning("⚠️ **Note:** This app is currently in the development phase and supports predictions for only a limited set of stocks.")
st.page_link("https://finwisely.org/stock/AAPL", label="🔙 Back to Main Page")
st.title("📈 Stock Price Prediction")

# Ticker selection
ticker_dict = {
    "Apple Inc (AAPL)": "AAPL",
    "Microsoft Corp (MSFT)": "MSFT",
    "Alphabet Inc (GOOGL)": "GOOGL",
    "Amazon.com Inc (AMZN)": "AMZN",
    "NVIDIA Corp (NVDA)": "NVDA",
    "Meta Platforms Inc (META)": "META",
    "Tesla Inc (TSLA)": "TSLA",
    "Broadcom Inc (AVGO)": "AVGO",
    "PepsiCo Inc (PEP)": "PEP",
    "Costco Wholesale Corp (COST)": "COST",
    "Cisco Systems Inc (CSCO)": "CSCO",
    "Adobe Inc (ADBE)": "ADBE",
    "Texas Instruments Inc (TXN)": "TXN",
    "Intel Corp (INTC)": "INTC",
    "Advanced Micro Devices Inc (AMD)": "AMD",
    "Starbucks Corp (SBUX)": "SBUX",
    "Charter Communications Inc (CHTR)": "CHTR",
    "Intuit Inc (INTU)": "INTU",
    "Booking Holdings Inc (BKNG)": "BKNG",
    "Moderna Inc (MRNA)": "MRNA"
}
selected_company = st.selectbox("Select a Stock:", list(ticker_dict.keys()))
ticker = ticker_dict[selected_company]

target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100

st.write(f"📈 Selected Ticker: **{selected_company}**")

if st.button("Predict"):
    result = get_stock_forecast_model(ticker, target_price, target_range)
    if result:
        data, model_pred, hit_month, hit_msg = result
        
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        
        st.subheader("🔮 Stock Price Forecast using SARIMA")
        st.write(model_pred)

        if hit_month:
            st.success(hit_msg)
        else:
            st.warning(hit_msg)

        # Forecast Graph
        st.subheader("📉 Forecast Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data, label="Actual Price", color="blue")
        ax.plot(pd.to_datetime(model_pred["Date"]), model_pred["Predicted Price"], label="Predicted Price", color="red", linestyle="dashed")
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.legend()
        st.pyplot(fig)
