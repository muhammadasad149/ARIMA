# import streamlit as st
# import pandas as pd
# import yfinance as yf
# from datetime import date, timedelta
# import matplotlib.pyplot as plt
# import numpy as np
# import logging
# import pickle
# import os

# # Configure logger
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def get_monthly_stock_data(ticker, start_date, end_date):
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
#     return stock_data.resample('ME').last()['Close']

# def get_forecast(ticker, max_months=60):
#     logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
#     start_date = "2023-01-01"
#     end_date = date.today().strftime("%Y-%m-%d")
#     data = get_monthly_stock_data(ticker, start_date, end_date)
#     if data.empty:
#         logger.error("Invalid ticker or no data found!")
#         return None, None
#     logger.info(f"Successfully fetched data. Records: {len(data)}")
    
#     model_filename = f"sarima/{ticker}.pkl"
#     if not os.path.exists(model_filename):
#         logger.error(f"Model file {model_filename} not found!")
#         return data, None
#     try:
#         with open(model_filename, "rb") as f:
#             model = pickle.load(f)
#         logger.info("SARIMA model loaded successfully.")
#     except Exception as e:
#         logger.error(f"Error loading SARIMA model: {e}")
#         return data, None
#     try:
#         forecast = model.forecast(steps=max_months)
#         logger.info("SARIMA forecast generated.")
#     except Exception as e:
#         logger.error(f"SARIMA forecast failed: {e}")
#         return data, None

#     future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, max_months + 1)]
#     pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
#     return data, pred_df

# def check_target_hit(pred_df, target_price, target_range=0.05):
#     # Identify the first month when the predicted price is within target range.
#     hit_index = (pred_df["Predicted Price"] >= target_price * (1 - target_range)) & (pred_df["Predicted Price"] <= target_price * (1 + target_range))
#     if hit_index.any():
#         # np.argmax gives the first index where True occurs; index +1 for month count.
#         hit_month = np.argmax(hit_index.values) + 1  
#         hit_month_years = round(hit_month / 12, 2)
#         hit_msg = f"🎯 Target price is likely to be reached in **{hit_month} months / {hit_month_years} years**."
#     else:
#         hit_month = None
#         hit_msg = "⚠️ Target price may not be reached within 5 years."
#     return hit_month, hit_msg

# # ------------------------------
# # STREAMLIT UI
# # ------------------------------

# st.set_page_config(page_title="Financial Consultant", page_icon="📈")
# st.warning("⚠️ **Note:** This app is currently in the development phase and supports predictions for only a limited set of stocks.")
# st.page_link("https://finwisely.org/stock/AAPL", label="🔙 Back to Main Page")
# st.title("📈 Stock Price Prediction")

# # Ticker selection
# ticker_dict = {
#     "Apple Inc (AAPL)": "AAPL",
#     "Microsoft Corp (MSFT)": "MSFT",
#     "Alphabet Inc (GOOGL)": "GOOGL",
#     "Amazon.com Inc (AMZN)": "AMZN",
#     "NVIDIA Corp (NVDA)": "NVDA",
#     "Meta Platforms Inc (META)": "META",
#     "Tesla Inc (TSLA)": "TSLA",
#     "Broadcom Inc (AVGO)": "AVGO",
#     "PepsiCo Inc (PEP)": "PEP",
#     "Costco Wholesale Corp (COST)": "COST",
#     "Cisco Systems Inc (CSCO)": "CSCO",
#     "Adobe Inc (ADBE)": "ADBE",
#     "Texas Instruments Inc (TXN)": "TXN",
#     "Intel Corp (INTC)": "INTC",
#     "Advanced Micro Devices Inc (AMD)": "AMD",
#     "Starbucks Corp (SBUX)": "SBUX",
#     "Charter Communications Inc (CHTR)": "CHTR",
#     "Intuit Inc (INTU)": "INTU",
#     "Booking Holdings Inc (BKNG)": "BKNG",
#     "Moderna Inc (MRNA)": "MRNA"
# }
# selected_company = st.selectbox("Select a Stock:", list(ticker_dict.keys()))
# ticker = ticker_dict[selected_company]

# # Prediction type selection
# prediction_option = st.radio("Select Prediction Option:", ("Price Target", "Months", "Both"))

# # Input fields based on selected option
# target_price = None
# months_input = None
# target_range = 0.05  # default target range

# if prediction_option in ("Price Target", "Both"):
#     target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
#     target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100

# if prediction_option in ("Months", "Both"):
#     months_input = st.number_input("Enter Number of Months for Prediction:", min_value=1, max_value=60, value=30)

# st.write(f"📈 Selected Ticker: **{selected_company}**")

# if st.button("Predict"):
#     data, pred_df = get_forecast(ticker, max_months=60)
#     if pred_df is None:
#         st.error("Forecast could not be generated.")
#     else:
#         st.subheader("📊 Recent Stock Prices")
#         st.write(data.tail(5))
#         st.subheader("🔮 Stock Price Forecast using SARIMA")
#         st.write(pred_df)

#         messages = []
#         # Process Months option: Show forecasted price at the given month.
#         if prediction_option in ("Months", "Both") and months_input:
#             if months_input <= len(pred_df):
#                 forecasted_price = pred_df.loc[months_input - 1, "Predicted Price"]
#                 messages.append(f"📅 Predicted price at **{months_input} months** is **${forecasted_price:.2f}**.")
#             else:
#                 messages.append("⚠️ Months input exceeds forecast range.")
        
#         # Process Price Target option: Show when target price is hit.
#         if prediction_option in ("Price Target", "Both") and target_price:
#             hit_month, hit_msg = check_target_hit(pred_df, target_price, target_range)
#             messages.append(hit_msg)
        
#         # Display messages
#         for msg in messages:
#             if "⚠️" in msg:
#                 st.warning(msg)
#             else:
#                 st.success(msg)
        
#         # Plot forecast graph
#         st.subheader("📉 Forecast Graph")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(data.index, data, label="Actual Price", color="blue")
#         ax.plot(pd.to_datetime(pred_df["Date"]), pred_df["Predicted Price"], label="Predicted Price", color="red", linestyle="dashed")
#         if prediction_option in ("Price Target", "Both") and target_price:
#             ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
#         ax.legend()
#         st.pyplot(fig)


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

def get_forecast(ticker, max_months=60):
    logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    data = get_monthly_stock_data(ticker, start_date, end_date)
    if data.empty:
        logger.error("Invalid ticker or no data found!")
        return None, None
    logger.info(f"Successfully fetched data. Records: {len(data)}")
    
    model_filename = f"sarima/{ticker}.pkl"
    if not os.path.exists(model_filename):
        logger.error(f"Model file {model_filename} not found!")
        return data, None
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        logger.info("SARIMA model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SARIMA model: {e}")
        return data, None
    try:
        forecast = model.forecast(steps=max_months)
        logger.info("SARIMA forecast generated.")
    except Exception as e:
        logger.error(f"SARIMA forecast failed: {e}")
        return data, None

    future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, max_months + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
    # Ensure the DataFrame has a RangeIndex for integer indexing
    pred_df.reset_index(drop=True, inplace=True)

 
    return data, pred_df 

def check_target_hit(pred_df, target_price, target_range=0.05):
    # Identify the first month when the predicted price is within target range.
    hit_index = (pred_df["Predicted Price"] >= target_price * (1 - target_range)) & (pred_df["Predicted Price"] <= target_price * (1 + target_range))
    if hit_index.any():
        hit_month = np.argmax(hit_index.values) + 1  # Index starts from 0, so add 1
        hit_month_years = round(hit_month / 12, 2)
        hit_msg = f"🎯 Target price is likely to be reached in **{hit_month} months / {hit_month_years} years**."
    else:
        hit_month = None
        hit_msg = "⚠️ Target price may not be reached within 5 years."
    return hit_month, hit_msg

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

# Prediction type selection
prediction_option = st.radio("Select Prediction Option:", ("Price Target", "Months", "Both"))

# Input fields based on selected option
target_price = None
months_input = None
if prediction_option in ("Price Target", "Both"):
    target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
    target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100
if prediction_option in ("Months", "Both"):
    months_input = st.number_input("Enter Number of Months for Prediction:", min_value=1, max_value=60, value=30)

st.write(f"📈 Selected Ticker: **{selected_company}**")

if st.button("Predict"):
    data, pred_df = get_forecast(ticker, max_months=60)
    if pred_df is None:
        st.error("Forecast could not be generated.")
    else:
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        st.subheader("🔮 Stock Price Forecast using SARIMA")
        st.write(pred_df)

        messages = []
        # Process Months option: Show forecasted price at the given month.
        if prediction_option in ("Months", "Both") and months_input:
            if months_input <= len(pred_df):
                # Using .iloc for integer-based indexing
                forecasted_price = pred_df.iloc[months_input - 1]["Predicted Price"]
                messages.append(f"📅 Predicted price at **{months_input} months** is **${forecasted_price:.2f}**.")
            else:
                messages.append("⚠️ Months input exceeds forecast range.")
        
        # Process Price Target option: Show when target price is hit.
        if prediction_option in ("Price Target", "Both") and target_price:
            hit_month, hit_msg = check_target_hit(pred_df, target_price, target_range)
            messages.append(hit_msg)
        
        # Display messages
        for msg in messages:
            if "⚠️" in msg:
                st.warning(msg)
            else:
                st.success(msg)
        
        # Plot forecast graph
        st.subheader("📉 Forecast Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data, label="Actual Price", color="blue")
        ax.plot(pd.to_datetime(pred_df["Date"]), pred_df["Predicted Price"], label="Predicted Price", color="red", linestyle="dashed")
        if prediction_option in ("Price Target", "Both") and target_price:
            ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.legend()
        st.pyplot(fig)

        
    
