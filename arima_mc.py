import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo Simulation Function
def monte_carlo_simulation(last_price, days, mu, sigma, target_price, simulations=10000):
    np.random.seed(42)
    paths = np.zeros((simulations, days))
    hit_days = []
    
    for i in range(simulations):
        prices = [last_price]
        for _ in range(days):
            price = prices[-1] * np.exp(mu + sigma * np.random.normal())
            prices.append(price)
        paths[i] = prices[1:]
        
        # Check when the target price is hit
        for j, price in enumerate(prices[1:]):
            if price >= target_price:
                hit_days.append(j + 1)
                break
    
    avg_hit_day = np.mean(hit_days) if hit_days else None
    return paths, avg_hit_day

# Function to fetch stock data and predict
def get_stock_forecast(ticker, days, target_price):
    try:
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = "2023-01-01"

        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("Invalid ticker or no data found!")
            return None

        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data.dropna(subset=["Close"], inplace=True)

        # Fit ARIMA model
        arima_model = ARIMA(data["Close"], order=(30, 1, 5))
        arima_result = arima_model.fit()
        
        arima_forecast = arima_result.get_forecast(steps=days)
        arima_pred = arima_forecast.summary_frame()

        # Compute drift and volatility for Monte Carlo
        log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()

        # Monte Carlo Simulation
        paths, avg_hit_day = monte_carlo_simulation(data["Close"].iloc[-1], days, mu, sigma, target_price)

        return data, arima_pred, avg_hit_day, paths

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title("📈 Stock Price Prediction (ARIMA + Monte Carlo)")

ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
days = st.number_input("Enter Number of Days for Prediction:", min_value=1, max_value=365, value=30)
target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)

if st.button("Predict"):
    result = get_stock_forecast(ticker, days, target_price)
    
    if result:
        data, arima_pred, avg_hit_day, paths = result
        
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        
        st.subheader(f"🔮 {days}-Day Stock Price Prediction")
        st.write(arima_pred)
        
        if avg_hit_day:
            st.success(f"🎯 Based on Monte Carlo Simulation, the target price **${target_price}** is expected to be hit in **{avg_hit_day:.2f} days** on average.")
        else:
            st.warning("⚠️ Target price may not be reached within the given days.")

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
        
        # Plot Monte Carlo Simulations
        st.subheader("📉 Monte Carlo Simulation Paths")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, days + 1), paths[:100].T, color='gray', alpha=0.1)  # Plot 100 sample paths
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Monte Carlo Simulated Stock Price Paths")
        ax.legend()
        st.pyplot(fig)
