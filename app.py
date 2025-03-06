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
# from openai import OpenAI
import io
# from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load environment variables from .env file
# load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# FUNCTIONS
# ------------------------------

def monte_carlo_simulation(last_price, days, mu, sigma, target_price, target_range=0.05, simulations=10000):
    np.random.seed(42)
    paths = []  # Using a list for simulation paths
    hit_days = []
    
    # Ensure parameters are floats
    mu = float(mu)
    sigma = float(sigma)
    last_price = float(last_price)
    
    for i in range(simulations):
        prices = [last_price]
        for _ in range(days):
            next_price = float(prices[-1]) * np.exp(mu + sigma * np.random.normal())
            prices.append(next_price)
        paths.append(prices[1:])  # Append the simulation path

        # Check when the price enters the target range (target ± target_range)
        for j, p in enumerate(prices[1:]):
            if target_price * (1 - target_range) <= float(p) <= target_price * (1 + target_range):
                hit_days.append(j + 1)  # Record the day when it enters the range
                break

    avg_hit_day = np.mean(hit_days) if hit_days else None
    hit_count = len(hit_days)
    return paths, avg_hit_day, hit_count

def get_monthly_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.resample('ME').last()['Close']

# def get_image_description(image_buffer):
#     # Convert image buffer to base64
#     encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Describe the following image in detail. Analyze the stock chart, including Bollinger Bands, RSI, and MACD indicators. Evaluate trends, momentum, and potential reversal signals. Provide a final investment recommendation."},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
#                 ],
#             }
#         ],
#     )
#     return response.choices[0].message.content

def get_stock_forecast_model(ticker, days, target_price, target_range=0.05):
    logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    
    # Fetch monthly stock data
    data = get_monthly_stock_data(ticker, start_date, end_date)
    if data.empty:
        logger.error("Invalid ticker or no data found!")
        return None
    logger.info(f"Successfully fetched data. Records: {len(data)}")
    
    # if model_type == "ARIMA":
    #     model_filename = f"arima/{ticker}.pkl"
    #     if not os.path.exists(model_filename):
    #         logger.error(f"Model file {model_filename} not found!")
    #         return None
    #     try:
    #         with open(model_filename, "rb") as f:
    #             model = pickle.load(f)
    #         logger.info("ARIMA model loaded successfully.")
    #     except Exception as e:
    #         logger.error(f"Error loading ARIMA model: {e}")
    #         return None
        
    #     # Forecast using ARIMA
    #     try:
    #         forecast = model.forecast(steps=days)
    #         logger.info("ARIMA forecast generated.")
    #     except Exception as e:
    #         logger.error(f"ARIMA forecast failed: {e}")
    #         return None
        
    #     future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, days + 1)]
    #     pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
    #     hit = (forecast > target_price * (1 - target_range)) & (forecast < target_price * (1 + target_range))
    #     target_hit_msg = "🎯 Target price is likely to be reached!" if hit.any() else "⚠️ Target price may not be reached."
    #     model_pred = pred_df
    
    # elif model_type == "SARIMA":
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
        forecast = model.forecast(steps=days)
        logger.info("SARIMA forecast generated.")
    except Exception as e:
        logger.error(f"SARIMA forecast failed: {e}")
        return None
    
    future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, days + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
    hit = (forecast > target_price * (1 - target_range)) & (forecast < target_price * (1 + target_range))
    target_hit_msg = "🎯 Target price is likely to be reached!" if hit.any() else "⚠️ Target price may not be reached."
    model_pred = pred_df
    
    # elif model_type == "LSTM":
    #     model_filename = f"lstm/{ticker}.h5"
    #     if not os.path.exists(model_filename):
    #         logger.error(f"LSTM model file {model_filename} not found!")
    #         return None
    #     try:
    #         lstm_model = load_model(model_filename)
    #         logger.info("LSTM model loaded successfully.")
    #     except Exception as e:
    #         logger.error(f"Error loading LSTM model: {e}")
    #         return None
        
    #     # Use the last 15 months of data to predict the next 'days' values
    #     closing_prices = data.values[-15:].reshape(-1, 1)
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     closing_prices_scaled = scaler.fit_transform(closing_prices)
    #     # Reshape to (1, 1, 15) as expected by the LSTM model
    #     input_features = closing_prices_scaled.reshape(1, 1, 15)
    #     predictions = []
    #     for i in range(days):
    #         predicted_scaled = lstm_model.predict(input_features)
    #         predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]
    #         predictions.append(predicted_price)
    #         # Update input_features by rolling and appending the new prediction
    #         input_features = np.roll(input_features, shift=-1, axis=2)
    #         input_features[0, 0, -1] = predicted_scaled[0, 0]
        
    #     future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, days + 1)]
    #     pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
    #     hit_array = (np.array(predictions) > target_price * (1 - target_range)) & (np.array(predictions) < target_price * (1 + target_range))
    #     target_hit_msg = "🎯 Target price is likely to be reached!" if hit_array.any() else "⚠️ Target price may not be reached."
    #     model_pred = pred_df
        
    # else:
    #     logger.error("Invalid model type selected!")
    #     return None
    
    # Monte Carlo Simulation (common for all models)
    log_returns = np.log(data / data.shift(1)).dropna()
    mu, sigma = float(log_returns.mean()), float(log_returns.std())
    last_price_val = data.iloc[-1]
    if isinstance(last_price_val, pd.Series):
        last_price_val = last_price_val.iloc[0]
    last_price_val = float(last_price_val)
    paths, avg_hit_day, hit_count = monte_carlo_simulation(last_price_val, days, mu, sigma, target_price, target_range)
    
    return data, model_pred, avg_hit_day, paths, hit_count, target_hit_msg

# ------------------------------
# STREAMLIT UI
# ------------------------------

st.set_page_config(page_title="Financial Consultant", page_icon="svgviewer-output.png")

st.warning("⚠️ **Note:** This app is currently in the development phase and currently supports predictions for only a limited set of stocks.")
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

# # Model selection dropdown
# model_options = ["ARIMA", "SARIMA", "LSTM"]
# selected_model = st.selectbox("Select Forecasting Model:", model_options)

days = st.number_input("Enter Number of Months for Prediction:", min_value=1, max_value=365, value=30)
target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100

# st.write(f"📈 Selected Ticker: **{selected_company}** | Forecasting Model: **{selected_model}**")
st.write(f"📈 Selected Ticker: **{selected_company}**")

if "bollinger_fig" not in st.session_state:
    st.session_state.bollinger_fig = None

if st.button("Predict"):
    result = get_stock_forecast_model(ticker, days, target_price, target_range)
    if result:
        data, model_pred, avg_hit_day, paths, hit_count, target_hit_msg = result
        
        st.subheader("📊 Recent Stock Prices")
        st.write(data.tail(5))
        
        st.subheader(f"🔮 {days}-Months Stock Price Prediction using SARIMA")
        st.write(model_pred)
        
        if "not" not in target_hit_msg.strip():
            st.success("🎯 Target price is likely to be reached within the given days.")
        else:
            st.warning("⚠️ Target price may not be reached within the forecasted period.")
        
        
        # Forecast Graph
        st.subheader("📉 Forecast Graph")
        fig1, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data, label="Actual Price", color="blue")
        ax.plot(pd.to_datetime(model_pred["Date"]), model_pred["Predicted Price"], label="Predicted Price", color="red", linestyle="dashed")
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.legend()
        st.pyplot(fig1)

        if avg_hit_day:
            st.success(f"🎯 Monte Carlo Simulation indicates an average hit Months of **{avg_hit_day:.2f} Months** with a probability of **{hit_count/10000:.2%}**.")
        else:
            st.warning("⚠️ Target price may not be reached within the given days (Monte Carlo Simulation).")

        
        # Monte Carlo Simulation Paths Graph
        st.subheader("📉 Monte Carlo Simulation Paths")
        fig3, ax = plt.subplots(figsize=(10, 5))
        paths_arr = np.array(paths)  # Convert list to NumPy array
        ax.plot(range(1, days + 1), paths_arr[:100].T, color='gray', alpha=0.1)
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Monte Carlo Simulated Stock Price Paths")
        ax.legend()
        st.pyplot(fig3)

        # # Compute technical indicators
        # def compute_rsi(series, period=14):
        #     delta = series.diff()
        #     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        #     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        #     rs = gain / loss
        #     return 100 - (100 / (1 + rs))

        # rsi = compute_rsi(data)
        # ema_12 = data.ewm(span=12, adjust=False).mean()
        # ema_26 = data.ewm(span=26, adjust=False).mean()
        # macd = ema_12 - ema_26
        # signal = macd.ewm(span=9, adjust=False).mean()
        # hist = macd - signal

        # window = 20
        # sma_20 = data.rolling(window=window).mean()
        # std_dev = data.rolling(window=window).std()
        # upper_band = sma_20 + (std_dev * 2)
        # lower_band = sma_20 - (std_dev * 2)

        # # Flatten arrays to ensure they are 1D
        # upper_band_arr = np.ravel(upper_band.values)
        # lower_band_arr = np.ravel(lower_band.values)
        # sma_20_arr = np.ravel(sma_20.values)
        # hist_arr = np.ravel(hist.values)

        # st.subheader("📉 Stock Price with Bollinger Bands and Forecast")
        # fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # # Top subplot: Price with Bollinger Bands
        # axes[0].plot(data.index, data, label='Actual Price', color='blue')
        # axes[0].plot(pd.to_datetime(model_pred["Date"]), model_pred['Predicted Price'], 
        #             label='Predicted Price', color='red', linestyle='dashed')
        # axes[0].fill_between(
        #     pd.to_datetime(model_pred["Date"]), 
        #     model_pred.get('mean_ci_lower', model_pred["Predicted Price"] - 5), 
        #     model_pred.get('mean_ci_upper', model_pred["Predicted Price"] + 5), 
        #     color='pink', alpha=0.3
        # )
        # axes[0].plot(data.index, sma_20_arr, label='20-day SMA', color='orange')
        # axes[0].fill_between(
        #     data.index, 
        #     upper_band_arr, 
        #     lower_band_arr, 
        #     color='gray', alpha=0.3, 
        #     label='Bollinger Bands'
        # )
        # axes[0].axhline(y=target_price, color='green', linestyle='--', 
        #                 label=f'Target Price: ${target_price}')
        # axes[0].legend()
        # axes[0].set_title("Stock Price with Bollinger Bands and Forecast")

        # # Middle subplot: RSI
        # axes[1].plot(data.index, rsi, label='RSI', color='purple')
        # axes[1].axhline(y=70, color='red', linestyle='dashed', label='Overbought (70)')
        # axes[1].axhline(y=30, color='green', linestyle='dashed', label='Oversold (30)')
        # axes[1].set_ylim(0, 100)
        # axes[1].legend()
        # axes[1].set_title("Relative Strength Index (RSI)")

        # # Bottom subplot: MACD
        # axes[2].plot(data.index, macd, label='MACD', color='blue')
        # axes[2].plot(data.index, signal, label='Signal Line', color='red')
        # axes[2].bar(data.index, hist_arr, label='Histogram', color='gray', alpha=0.5)
        # axes[2].legend()
        # axes[2].set_title("MACD (Moving Average Convergence Divergence)")

        # plt.tight_layout()
        # st.session_state.bollinger_fig = fig
        # st.pyplot(fig)


        # with st.spinner("Analyzing the Stock Price with Bollinger Bands and Forecast graphs"):
        #     try:
        #         if st.session_state.bollinger_fig is None:
        #             st.sidebar.error("Figure is missing from session state.")
        #         else:
        #             fig_buffer = io.BytesIO()
        #             st.session_state.bollinger_fig.savefig(fig_buffer, format="png")
        #             fig_buffer.seek(0)
        #             description = get_image_description(fig_buffer)
        #             st.sidebar.subheader("Image Insights")
        #             st.sidebar.write(description)
        #     except Exception as e:
        #         st.sidebar.error(f"Error in generating insights: {e}")


