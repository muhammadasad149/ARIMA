import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt


# Modified Monte Carlo Simulation Function with Range Handling
def monte_carlo_simulation(last_price, days, mu, sigma, target_price, target_range=0.05, simulations=10000):
    np.random.seed(42)
    paths = np.zeros((simulations, days))
    hit_days = []
    
    for i in range(simulations):
        prices = [last_price]
        for _ in range(days):
            price = prices[-1] * np.exp(mu + sigma * np.random.normal())
            prices.append(price)
        paths[i] = prices[1:]
        
        # Check when the price enters the target range (target ± target_range)
        for j, price in enumerate(prices[1:]):
            if target_price * (1 - target_range) <= price <= target_price * (1 + target_range):
                hit_days.append(j + 1)  # Record the day when it enters the range
                break
    
    avg_hit_day = np.mean(hit_days) if hit_days else None
    hit_count = len(hit_days)  # Count how many times the target range was hit
    return paths, avg_hit_day, hit_count

# Function to fetch stock data and predict with updated Monte Carlo
def get_stock_forecast(ticker, days, target_price, target_range=0.05 , p=30 , d= 0 , q=10):
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
        arima_model = ARIMA(data["Close"], order=(p,d,q))
        arima_result = arima_model.fit()
        
        arima_forecast = arima_result.get_forecast(steps=days)
        arima_pred = arima_forecast.summary_frame()

        # Check when the target price will be hit
        arima_target_hit_day = None
        for i, price in enumerate(arima_pred["mean"]):
            if price >= target_price:
                arima_target_hit_day = (date.today() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                break

        # Compute drift and volatility for Monte Carlo
        log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std()

        # Monte Carlo Simulation
        paths, avg_hit_day, hit_count = monte_carlo_simulation(data["Close"].iloc[-1], days, mu, sigma, target_price, target_range)

        return data, arima_pred, avg_hit_day, paths, hit_count , arima_target_hit_day

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI (updated)
st.title("📈 Stock Price Prediction (ARIMA + Monte Carlo)")

ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
days = st.number_input("Enter Number of Days for Prediction:", min_value=1, max_value=365, value=30)
target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100  # Range as a percentage
p = st.number_input("Enter ARIMA Order (p):", min_value=0, value=30, step=1)
d = st.number_input("Enter ARIMA Differencing Order (d):", min_value=0, value=1, step=1)
q = st.number_input("Enter ARIMA Order (q):", min_value=0, value=10, step=1)

# Add the explanation in the sidebar
st.sidebar.subheader("ARIMA Model Parameters Explanation")

st.sidebar.write("""
- **p**: **AR (Auto-Regressive) order** - This represents the number of lag observations (previous time steps) that are used in the model. It helps in capturing the relationship between the current value and previous values in the time series.

- **d**: **Differencing order** - This is the number of times the data is differenced to make the time series stationary. Differencing removes trends and seasonality from the data, making it more predictable.

- **q**: **MA (Moving Average) order** - This represents the number of past forecast errors used to make the model more stable and accurate. It helps in correcting the past forecast errors to improve the prediction.

These parameters together define the ARIMA model's structure:
- **p** controls the autoregressive part,
- **d** controls the differencing to stabilize the data,
- **q** controls the moving average to correct past errors.
""")

if st.button("Predict"):
    result = get_stock_forecast(ticker, days, target_price, target_range , p, d, q)
    
    if result:
        data, arima_pred, avg_hit_day, paths, hit_count , arima_target_hit_day = result
        
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        
        st.subheader(f"🔮 {days}-Day Stock Price Prediction")
        st.write(arima_pred)

        # Show target price prediction
        if arima_target_hit_day:
            st.success(f"🎯 Target price **${target_price}** will be hit on **{arima_target_hit_day}**.")
        else:
            st.warning("⚠️ Target price **may not be reached** within the given days.")
        
        if avg_hit_day:
            st.success(f"🎯 Based on Monte Carlo Simulation, the target price **${target_price}** is expected to be hit in **{avg_hit_day:.2f} days** on average.")
            if hit_count > 1:
                st.success(f"💥 The target price range was hit **{hit_count}** times in the simulation.")
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
