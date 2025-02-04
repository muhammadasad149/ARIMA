# import streamlit as st
# import pandas as pd
# import yfinance as yf
# from statsmodels.tsa.arima.model import ARIMA
# from datetime import date
# import matplotlib.pyplot as plt

# # Function to fetch stock data and predict
# def get_stock_forecast(ticker):
#     try:
#         end_date = date.today().strftime("%Y-%m-%d")
#         start_date = "2023-01-01"

#         # Data fetch karein
#         data = yf.download(ticker, start=start_date, end=end_date)

#         # Check if data is empty
#         if data.empty:
#             st.error("Invalid ticker or no data found!")
#             return None

#         # Ensure 'Close' column is numeric
#         data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
#         data.dropna(subset=["Close"], inplace=True)

#         # Fit ARIMA model
#         arima_model = ARIMA(data["Close"], order=(1, 1, 1))
#         arima_result = arima_model.fit()

#         # Forecast for 30 days
#         arima_forecast = arima_result.get_forecast(steps=30)
#         arima_pred = arima_forecast.summary_frame()

#         return data, arima_pred

#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None

# # Streamlit UI
# st.title("📈 Stock Price Prediction (ARIMA)")

# # User input for ticker
# ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()

# if st.button("Predict"):
#     result = get_stock_forecast(ticker)
    
#     if result:
#         data, arima_pred = result

#         # Show last few stock prices
#         st.subheader("📊 Recent Stock Prices")
#         st.write(data.tail(5))

#         # Show forecast data
#         st.subheader("🔮 30-Day Stock Price Prediction")
#         st.write(arima_pred)

#         # Plot prediction
#         st.subheader("📉 Forecast Graph")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(data.index, data["Close"], label="Actual Price", color="blue")
#         ax.plot(pd.date_range(start=data.index[-1], periods=31, freq='D')[1:], 
#                 arima_pred["mean"], label="Predicted Price", color="red", linestyle="dashed")
#         ax.fill_between(pd.date_range(start=data.index[-1], periods=31, freq='D')[1:], 
#                         arima_pred["mean_ci_lower"], arima_pred["mean_ci_upper"], color="pink", alpha=0.3)
#         ax.legend()
#         st.pyplot(fig)


import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import matplotlib.pyplot as plt

# Function to fetch stock data and predict
def get_stock_forecast(ticker, days, target_price):
    try:
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = "2023-01-01"

        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check if data is empty
        if data.empty:
            st.error("Invalid ticker or no data found!")
            return None

        # Ensure 'Close' column is numeric
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data.dropna(subset=["Close"], inplace=True)

        # Fit ARIMA model
        arima_model = ARIMA(data["Close"], order=(30, 0, 0))
        arima_result = arima_model.fit()

        # Forecast for user-specified days
        arima_forecast = arima_result.get_forecast(steps=days)
        arima_pred = arima_forecast.summary_frame()

        # Check when the target price will be hit
        target_hit_day = None
        for i, price in enumerate(arima_pred["mean"]):
            if price >= target_price:
                target_hit_day = (date.today() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                break

        return data, arima_pred, target_hit_day

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title("📈 Stock Price Prediction (ARIMA)")

# User inputs
ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
days = st.number_input("Enter Number of Days for Prediction:", min_value=1, max_value=365, value=30)
target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)

if st.button("Predict"):
    result = get_stock_forecast(ticker, days, target_price)
    
    if result:
        data, arima_pred, target_hit_day = result

        # Show last few stock prices
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))

        # Show forecast data
        st.subheader(f"🔮 {days}-Day Stock Price Prediction")
        st.write(arima_pred)

        # Show target price prediction
        if target_hit_day:
            st.success(f"🎯 Target price **${target_price}** will be hit on **{target_hit_day}**.")
        else:
            st.warning("⚠️ Target price **may not be reached** within the given days.")

        # Plot prediction
        st.subheader("📉 Forecast Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Close"], label="Actual Price", color="blue")
        ax.plot(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:], 
                arima_pred["mean"], label="Predicted Price", color="red", linestyle="dashed")
        ax.fill_between(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:], 
                        arima_pred["mean_ci_lower"], arima_pred["mean_ci_upper"], color="pink", alpha=0.3)
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.legend()
        st.pyplot(fig)
