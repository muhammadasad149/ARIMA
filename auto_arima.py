import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
import base64
import os
from openai import OpenAI
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def get_image_description(image_buffer):
    # Convert image buffer to base64
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the following image in detail. Analyze the stock chart, including Bollinger Bands, RSI, and MACD indicators. Evaluate trends, momentum, and potential reversal signals. Provide a final investment recommendation."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


# Function to Fetch Stock Data and Predict with ARIMA + Monte Carlo
def get_stock_forecast(ticker, days, target_price, target_range=0.05):
    logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")

    end_date = date.today().strftime("%Y-%m-%d")
    start_date = "2023-01-01"

    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        logger.error("Invalid ticker or no data found!")
        return None

    logger.info(f"Successfully fetched data. Records: {len(data)}")

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data.dropna(subset=["Close"], inplace=True)

    # Load ARIMA Model Dynamically
    model_filename = f"arima_models/{ticker}.pickle"
    if not os.path.exists(model_filename):
        logger.error(f"Model file {model_filename} not found!")
        return None

    try:
        with open(model_filename, "rb") as f:
            arima_model = pickle.load(f)
        logger.info("ARIMA model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

    # Forecast with ARIMA
    try:
        arima_forecast = arima_model.get_forecast(steps=days)
        arima_pred = arima_forecast.summary_frame()
        logger.info("ARIMA forecast generated.")
    except Exception as e:
        logger.error(f"ARIMA forecast failed: {e}")
        return None

    # Check if Target Price Falls in Forecast
    target_lower = target_price * (1 - target_range)
    target_upper = target_price * (1 + target_range)
    conf_int = arima_pred[["mean_ci_lower", "mean_ci_upper"]]

    hit = (conf_int.iloc[:, 0] <= target_upper) & (conf_int.iloc[:, 1] >= target_lower)
    arima_target_hit_day = "🎯 Target price is likely to be reached!" if hit.any() else "⚠️ Target price may not be reached."

    # Monte Carlo Simulation
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()
    paths, avg_hit_day, hit_count = monte_carlo_simulation(data["Close"].iloc[-1], days, mu, sigma, target_price, target_range)
    
    return data, arima_pred, avg_hit_day, paths, hit_count, arima_target_hit_day

# Change the browser tab title
st.set_page_config(page_title="Financial Consultant", page_icon="svgviewer-output.png")

st.page_link("https://finwisely.org/stock/AAPL", label="🔙 Back to Main Page")

# Streamlit UI (updated)
st.title("📈 Stock Price Prediction (ARIMA + Monte Carlo)")

# st.page_link("https://finwisely.org/stock/AAPL", label="🔙 Back to Main Page")

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

# Create dropdown for ticker selection
selected_company = st.selectbox("Select a Stock:", list(ticker_dict.keys()))

# Extract selected ticker
ticker = ticker_dict[selected_company]

days = st.number_input("Enter Number of Days for Prediction:", min_value=1, max_value=365, value=30)
target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100  # Range as a percentage

st.write(f"📈 Selected Ticker: **{selected_company}**")

if "bollinger_fig" not in st.session_state:
    st.session_state.bollinger_fig = None

if st.button("Predict"):
    result = get_stock_forecast(ticker , days, target_price, target_range)
    
    if result:
        data, arima_pred, avg_hit_day, paths, hit_count , arima_target_hit_day = result
        
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        
        st.subheader(f"🔮 {days}-Day Stock Price Prediction")
        st.write(arima_pred)

        check = arima_target_hit_day.strip()
        if "not" not in check:
            st.success("🎯 Target price **is likely to be reached** within the given days.")
        else:
            st.warning("⚠️ Target price **may not be reached** within the forecasted period.")
        
        if avg_hit_day:
            # st.success(f"🎯 Based on Monte Carlo Simulation, the target price **${target_price}** is expected to be hit in **{avg_hit_day:.2f} days** on average.")
            if hit_count > 1:
                # st.success(f"💥 The target price range was reached in {hit_count / 10000:.2%} of the simulations.")
                st.success(f"🎯 Based on Monte Carlo Simulation there is {hit_count / 10000:.2%} probability that {ticker} will hit the target price of **${target_price}**")
        else:
            st.warning("⚠️ Target price may not be reached within the given days.")
        
        # Plot prediction
        st.subheader("📉 Forecast Graph")
        fig1, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Close"], label="Actual Price", color="blue")
        ax.plot(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:], 
                arima_pred["mean"], label="Predicted Price", color="red", linestyle="dashed")
        ax.fill_between(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:], 
                        arima_pred["mean_ci_lower"], arima_pred["mean_ci_upper"], color="pink", alpha=0.3)
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.legend()
        st.pyplot(fig1)

        # Compute RSI manually
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        rsi = compute_rsi(data['Close'])

        # Compute MACD manually
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        # Compute Bollinger Bands manually
        window = 20  # 20-day SMA
        sma_20 = data['Close'].rolling(window=window).mean()
        std_dev = data['Close'].rolling(window=window).std()
        upper_band = sma_20 + (std_dev * 2)
        lower_band = sma_20 - (std_dev * 2)

        st.subheader("📉 Stock Price with Bollinger Bands and Forecast")
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        # fig, ax = plt.subplots(figsize=(10, 5))

        # Top subplot: Price with Bollinger Bands
        axes[0].plot(data.index, data['Close'], label='Actual Price', color='blue')
        axes[0].plot(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:],
                    arima_pred['mean'], label='Predicted Price', color='red', linestyle='dashed')
        axes[0].fill_between(pd.date_range(start=data.index[-1], periods=days + 1, freq='D')[1:],
                            arima_pred['mean_ci_lower'], arima_pred['mean_ci_upper'], color='pink', alpha=0.3)
        axes[0].plot(data.index, sma_20, label='20-day SMA', color='orange')
        axes[0].fill_between(data.index, upper_band, lower_band, color='gray', alpha=0.3, label='Bollinger Bands')
        axes[0].axhline(y=target_price, color='green', linestyle='--', label=f'Target Price: ${target_price}')
        axes[0].legend()
        axes[0].set_title("Stock Price with Bollinger Bands and Forecast")

        # Middle subplot: RSI
        axes[1].plot(data.index, rsi, label='RSI', color='purple')
        axes[1].axhline(y=70, color='red', linestyle='dashed', label='Overbought (70)')
        axes[1].axhline(y=30, color='green', linestyle='dashed', label='Oversold (30)')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].set_title("Relative Strength Index (RSI)")

        # Bottom subplot: MACD
        axes[2].plot(data.index, macd, label='MACD', color='blue')
        axes[2].plot(data.index, signal, label='Signal Line', color='red')
        axes[2].bar(data.index, hist, label='Histogram', color='gray', alpha=0.5)
        axes[2].legend()
        axes[2].set_title("MACD (Moving Average Convergence Divergence)")


        # Adjust layout
        plt.tight_layout()

        # Save the figure to session state
        st.session_state.bollinger_fig = fig

        st.pyplot(fig)
       
        with st.spinner("Analyzing the Stock Price with Bollinger Bands and Forecast graphs"):
            try:
                print("checkpoint 1")
                # Debugging: Check if fig is correctly stored
                if st.session_state.bollinger_fig is None:
                    st.sidebar.error("Figure is missing from session state.")
                else:

                    # Convert figure to BytesIO
                    fig_buffer = io.BytesIO()
                    st.session_state.bollinger_fig.savefig(fig_buffer, format="png")
                    fig_buffer.seek(0)

                    # Get image description
                    description = get_image_description(fig_buffer)

                    st.sidebar.subheader("Image Insights")
                    st.sidebar.write(description)

            except Exception as e:
                st.sidebar.error(f"Error in generating insights: {e}")

        # Plot Monte Carlo Simulations
        st.subheader("📉 Monte Carlo Simulation Paths")
        fig3, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, days + 1), paths[:100].T, color='gray', alpha=0.1)  # Plot 100 sample paths
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Monte Carlo Simulated Stock Price Paths")
        ax.legend()
        st.pyplot(fig3)
