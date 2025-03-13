import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import logging
import pickle
import os
import base64
from openai import OpenAI
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  GPT Image Description
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
#  Fetch DAILY data instead of monthly
# -------------------------------------------------------------------
def get_daily_stock_data(ticker, start_date, end_date):
    """
    Download daily stock data from yfinance.
    Returns a DataFrame with columns: ['Open','High','Low','Close','Adj Close','Volume'].
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    return data

# -------------------------------------------------------------------
#  Compute technical indicators: RSI, Bollinger Bands, MACD
# -------------------------------------------------------------------
def compute_rsi(series, period=14):
    """
    Computes RSI (Relative Strength Index) over the given 'series' of prices.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, window=20):
    """
    Computes 20-day SMA and +/-2 standard deviation Bollinger Bands.
    """
    sma_20 = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = sma_20 + (2 * std_dev)
    lower_band = sma_20 - (2 * std_dev)
    return sma_20, upper_band, lower_band

def compute_macd(series, fast=12, slow=26, signal_period=9):
    """
    Computes MACD (Moving Average Convergence Divergence) and signal/histogram.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# # -------------------------------------------------------------------
# #  Monte Carlo Simulation
# # -------------------------------------------------------------------
# def monte_carlo_simulation(last_price, days, mu, sigma, target_price, target_range=0.05, simulations=10000):
#     np.random.seed(42)
#     paths = []  # Using a list for simulation paths
#     hit_days = []
    
#     # Ensure parameters are floats
#     mu = float(mu)
#     sigma = float(sigma)
#     last_price = float(last_price)
    
#     for i in range(simulations):
#         prices = [last_price]
#         for _ in range(days):
#             next_price = float(prices[-1]) * np.exp(mu + sigma * np.random.normal())
#             prices.append(next_price)
#         paths.append(prices[1:])  # Append the simulation path

#         # Check when the price enters the target range (target ± target_range)
#         for j, p in enumerate(prices[1:]):
#             if target_price * (1 - target_range) <= float(p) <= target_price * (1 + target_range):
#                 hit_days.append(j + 1)  # Record the day when it enters the range
#                 break

#     avg_hit_day = np.mean(hit_days) if hit_days else None
#     hit_count = len(hit_days)
#     return paths, avg_hit_day, hit_count



# # -------------------------------------------------------------------
# #  Fetch MONTHLY data
# # -------------------------------------------------------------------
# def get_monthly_stock_data(ticker, start_date, end_date):
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
#     # Using 'M' for month-end frequency
#     return stock_data.resample('M').last()['Close']


# # -------------------------------------------------------------------
# #  Get forecast
# # -------------------------------------------------------------------
# def get_forecast(ticker, max_months=60, days=180, target_price=None, target_range=0.05):
#     logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
#     start_date = "2023-01-01"
#     end_date = date.today().strftime("%Y-%m-%d")
#     data = get_monthly_stock_data(ticker, start_date, end_date)
#     if data.empty:
#         logger.error("Invalid ticker or no data found!")
#         return None, None, None, None, None
#     logger.info(f"Successfully fetched data. Records: {len(data)}")
    
#     model_filename = f"sarima/{ticker}.pkl"
#     if not os.path.exists(model_filename):
#         logger.error(f"Model file {model_filename} not found!")
#         return data, None, None, None, None
#     try:
#         with open(model_filename, "rb") as f:
#             model = pickle.load(f)
#         logger.info("SARIMA model loaded successfully.")
#     except Exception as e:
#         logger.error(f"Error loading SARIMA model: {e}")
#         return data, None, None, None, None
#     try:
#         forecast = model.forecast(steps=max_months)
#         logger.info("SARIMA forecast generated.")
#     except Exception as e:
#         logger.error(f"SARIMA forecast failed: {e}")
#         return data, None, None, None, None

#     future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, max_months + 1)]
#     pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
#     # Ensure the DataFrame has a RangeIndex for integer indexing
#     pred_df.reset_index(drop=True, inplace=True)

#     # Monte Carlo Simulation (only if target_price is provided)
#     if target_price is not None:
#         log_returns = np.log(data / data.shift(1)).dropna()
#         mu_val, sigma_val = float(log_returns.mean()), float(log_returns.std())
#         last_price_val = data.iloc[-1]
#         if isinstance(last_price_val, pd.Series):
#             last_price_val = last_price_val.iloc[0]
#         last_price_val = float(last_price_val)
#         paths, avg_hit_day, hit_count = monte_carlo_simulation(
#             last_price_val, days, mu_val, sigma_val, target_price, target_range
#         )
#     else:
#         paths, avg_hit_day, hit_count = None, None, None
    
#     return data, pred_df, paths, avg_hit_day, hit_count

# #----------------------------------------------------------------
# #  Check target hit
# #----------------------------------------------------------------
# def check_target_hit(pred_df, target_price, target_range=0.05):
#     # Identify the first month when the predicted price is within target range.
#     hit_index = (
#         (pred_df["Predicted Price"] >= target_price * (1 - target_range)) &
#         (pred_df["Predicted Price"] <= target_price * (1 + target_range))
#     )
#     if hit_index.any():
#         hit_month = np.argmax(hit_index.values) + 1  # Index starts from 0, so add 1
#         hit_month_years = round(hit_month / 12, 2)
#         hit_msg = f"🎯 Target price is likely to be reached in **{hit_month} months / {hit_month_years} years**."
#     else:
#         hit_month = None
#         hit_msg = "⚠️ Target price may not be reached within 5 years."
#     return hit_month, hit_msg

# # ------------------------------
# # STREAMLIT UI
# # ------------------------------

# st.set_page_config(page_title="Financial Consultant", page_icon="svgviewer-output.png")
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
# target_range=0.05
# target_price = None
# months_input = None
# if prediction_option in ("Price Target", "Both"):
#     target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
#     target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100
# if prediction_option in ("Months", "Both"):
#     months_input = st.number_input("Enter Number of Months for Prediction:", min_value=1, max_value=60, value=30)
#     target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100

# st.write(f"📈 Selected Ticker: **{selected_company}**")

# if "bollinger_fig" not in st.session_state:
#     st.session_state.bollinger_fig = None

# if st.button("Predict"):
#     # Determine simulation days based on input.
#     simulation_days = months_input * 30 if months_input else 180

#     data, pred_df, paths, avg_hit_day, hit_count = get_forecast(
#         ticker,
#         max_months=60,
#         days=simulation_days,
#         target_price=target_price,
#         target_range=target_range
#     )
    
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
#                 forecasted_price = pred_df.iloc[months_input - 1]["Predicted Price"]
#                 messages.append(
#                     f"📅 Predicted price at **{months_input} months** is **${forecasted_price:.2f}**."
#                 )
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


#         # Monte Carlo Simulation Paths Graph
#         if paths is not None and hit_count is not None and avg_hit_day is not None:
#             if avg_hit_day:
#                 st.success(
#                     f"🎯 Monte Carlo Simulation indicates an average hit time of **{avg_hit_day:.2f} Months** "
#                     f"with a probability of **{hit_count/10000:.2%}**."
#                 )
#             else:
#                 st.warning(
#                     "⚠️ Target price may not be reached within the given simulation period (Monte Carlo Simulation)."
#                 )
#             st.subheader("📉 Monte Carlo Simulation Paths")
#             fig3, ax = plt.subplots(figsize=(10, 5))
#             paths_arr = np.array(paths)
#             ax.plot(range(1, simulation_days + 1), paths_arr[:100].T, color='gray', alpha=0.1)
#             ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")
#             ax.set_xlabel("Days")
#             ax.set_ylabel("Price")
#             ax.set_title("Monte Carlo Simulated Stock Price Paths")
#             ax.set_ylim(0, 2000)
#             ax.legend()
#             st.pyplot(fig3)

# -------------------------------------------------------------------
#  Monte Carlo Simulation
# -------------------------------------------------------------------
def monte_carlo_simulation(last_price, days, mu, sigma, target_price, target_range=0.05, simulations=10000):
    np.random.seed(42)
    paths = []
    hit_days = []
    
    # Ensure parameters are floats
    mu = float(mu)
    sigma = float(sigma)
    last_price = float(last_price)
    
    for _ in range(simulations):
        prices = [last_price]
        for _day in range(days):
            next_price = prices[-1] * np.exp(mu + sigma * np.random.normal())
            prices.append(next_price)
        paths.append(prices[1:])  # store path minus the initial price

        # Check if/when target range is reached
        for j, p in enumerate(prices[1:]):
            if (target_price * (1 - target_range) <= p <= target_price * (1 + target_range)):
                hit_days.append(j + 1)  # day index + 1
                break

    avg_hit_day = np.mean(hit_days) if hit_days else None
    hit_count = len(hit_days)
    return paths, avg_hit_day, hit_count

# -------------------------------------------------------------------
#  Fetch MONTHLY data
# -------------------------------------------------------------------
def get_monthly_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.resample('M').last()['Close']  # month-end

# -------------------------------------------------------------------
#  Get forecast
# -------------------------------------------------------------------
def get_forecast(ticker, max_months=60, days=180, target_price=None, target_range=0.05):
    """
    If target_price is provided, Monte Carlo is run inside this function.
    Otherwise, it returns None for (paths, avg_hit_day, hit_count).
    """
    logger.info(f"Fetching stock data for {ticker} from 2023-01-01 to {date.today()}")
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    data = get_monthly_stock_data(ticker, start_date, end_date)
    if data.empty:
        logger.error("Invalid ticker or no data found!")
        return None, None, None, None, None
    logger.info(f"Successfully fetched data. Records: {len(data)}")
    
    model_filename = f"sarima/{ticker}.pkl"
    if not os.path.exists(model_filename):
        logger.error(f"Model file {model_filename} not found!")
        return data, None, None, None, None
    
    # Load SARIMA model
    try:
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        logger.info("SARIMA model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SARIMA model: {e}")
        return data, None, None, None, None
    
    # Forecast
    try:
        forecast = model.forecast(steps=max_months)
        logger.info("SARIMA forecast generated.")
    except Exception as e:
        logger.error(f"SARIMA forecast failed: {e}")
        return data, None, None, None, None

    # Build pred_df
    future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, max_months + 1)]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": forecast})
    pred_df.reset_index(drop=True, inplace=True)

    # Monte Carlo inside get_forecast ONLY if user gave a direct target_price
    if target_price is not None:
        log_returns = np.log(data / data.shift(1)).dropna()
        mu_val, sigma_val = float(log_returns.mean()), float(log_returns.std())
        last_price_val = float(data.iloc[-1])
        paths, avg_hit_day, hit_count = monte_carlo_simulation(
            last_price_val, days, mu_val, sigma_val,
            target_price, target_range
        )
    else:
        paths, avg_hit_day, hit_count = None, None, None
    
    return data, pred_df, paths, avg_hit_day, hit_count

#----------------------------------------------------------------
#  Check target hit
#----------------------------------------------------------------
def check_target_hit(pred_df, target_price, target_range=0.05):
    hit_index = (
        (pred_df["Predicted Price"] >= target_price * (1 - target_range)) &
        (pred_df["Predicted Price"] <= target_price * (1 + target_range))
    )
    if hit_index.any():
        hit_month = np.argmax(hit_index.values) + 1  # first True index + 1
        hit_month_years = round(hit_month / 12, 2)
        hit_msg = f"🎯 Target price is likely to be reached in **{hit_month} months / {hit_month_years} years**."
    else:
        hit_month = None
        hit_msg = "⚠️ Target price may not be reached within 5 years."
    return hit_month, hit_msg

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="Financial Consultant", page_icon="svgviewer-output.png")
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

# Because we may not define them inside the if-block, set defaults:
target_price = None
months_input = None
target_range = 0.05  # default

# If user picks Price Target or Both
if prediction_option in ("Price Target", "Both"):
    target_price = st.number_input("Enter Target Price:", min_value=1.0, value=200.0, step=0.1)
    target_range = st.number_input("Enter Target Price Range (in %):", min_value=0.0, value=5.0, step=0.1) / 100

# If user picks Months or Both
if prediction_option in ("Months", "Both"):
    months_input = st.number_input("Enter Number of Months for Prediction:", min_value=1, max_value=60, value=30)
    # Optionally let them specify a target range for MC if we do a derived target
    # (But it's optional. We can reuse the same target_range or define a new slider.)

st.write(f"📈 Selected Ticker: **{selected_company}**")

if st.button("Predict"):
    # Convert months to days for MC
    simulation_days = months_input * 30 if months_input else 180
    
    # 1) Get SARIMA forecast. This only does MC if a direct target_price is given.
    data, pred_df, paths, avg_hit_day, hit_count = get_forecast(
        ticker,
        max_months=60,
        days=simulation_days,
        target_price=target_price,     # might be None if user picks "Months" only
        target_range=target_range
    )
    
    if pred_df is None:
        st.error("Forecast could not be generated.")
        st.stop()

    st.subheader("📊 Recent Stock Prices")
    st.write(data.tail(5))

    st.subheader("🔮 Stock Price Forecast using SARIMA")
    st.write(pred_df)

    messages = []

    # ------------------------------------------------------
    # (A) If user selected "Months" or "Both", show forecasted price
    # ------------------------------------------------------
    forecasted_price = None
    if prediction_option in ("Months", "Both") and months_input:
        if months_input <= len(pred_df):
            forecasted_price = pred_df.iloc[months_input - 1]["Predicted Price"]
            messages.append(
                f"📅 Predicted price at **{months_input} months** is **${forecasted_price:.2f}**."
            )
        else:
            messages.append("⚠️ Months input exceeds forecast range.")

    # ------------------------------------------------------
    # (B) If user selected "Price Target" or "Both", show target-hit info
    # ------------------------------------------------------
    if prediction_option in ("Price Target", "Both") and target_price:
        hit_month, hit_msg = check_target_hit(pred_df, target_price, target_range)
        messages.append(hit_msg)

    # Display messages
    for msg in messages:
        if "⚠️" in msg:
            st.warning(msg)
        else:
            st.success(msg)

    # ------------------------------------------------------
    # Plot the SARIMA forecast
    # ------------------------------------------------------
    st.subheader("📉 Forecast Graph")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data, label="Actual Price", color="blue")
    ax.plot(pd.to_datetime(pred_df["Date"]), pred_df["Predicted Price"],
            label="Predicted Price", color="red", linestyle="dashed")

    # Only draw user-defined target line if user actually gave a target price
    if prediction_option in ("Price Target", "Both") and target_price:
        ax.axhline(y=target_price, color="green", linestyle="--", label=f"Target Price: ${target_price}")

    ax.legend()
    st.pyplot(fig)

    # ------------------------------------------------------
    # (C) If user did NOT give a target_price but selected "Months",
    #     we run a separate Monte Carlo with the forecasted price as the target
    # ------------------------------------------------------
    if prediction_option == "Months" and target_price is None and forecasted_price is not None:
        # (i) Compute mu, sigma from monthly data
        log_returns = np.log(data / data.shift(1)).dropna()
        mu_val, sigma_val = float(log_returns.mean()), float(log_returns.std())
        last_price_val = float(data.iloc[-1])

        # (ii) Run Monte Carlo using forecasted_price as the target
        mc_paths, mc_avg_hit_day, mc_hit_count = monte_carlo_simulation(
            last_price_val,
            simulation_days,
            mu_val,
            sigma_val,
            target_price=forecasted_price,   # derived target
            target_range=target_range
        )

        # (iii) Show results
        if mc_avg_hit_day:
            st.success(
                f"🎯 Monte Carlo indicates an average hit time "
                f"of **{mc_avg_hit_day:.2f} Months** with a probability of **{mc_hit_count/10000:.2%}** "
                f"for the derived target (${forecasted_price:.2f})."
            )
        else:
            st.warning(
                "⚠️ Derived target price may not be reached within the simulation period (Monte Carlo)."
            )

        # (iv) Plot Monte Carlo
        st.subheader("📉 Monte Carlo Simulation Paths (Months-only Derived Target)")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        mc_paths_arr = np.array(mc_paths)
        ax3.plot(range(1, simulation_days + 1), mc_paths_arr[:100].T, color='gray', alpha=0.1)
        ax3.axhline(y=forecasted_price, color="green", linestyle="--",
                    label=f"Target Price: ${forecasted_price:.2f}")
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Price")
        ax3.set_title("Monte Carlo Simulated Stock Price Paths (Derived Target)")
        ax3.set_ylim(0, 2000)
        ax3.set_xlim(0, 200)
        ax3.legend()
        st.pyplot(fig3)

    # ------------------------------------------------------
    # (D) If user provided an actual target_price ("Price Target" or "Both"),
    #     we already have MC results from get_forecast() → (paths, avg_hit_day, hit_count).
    # ------------------------------------------------------
    elif paths is not None and avg_hit_day is not None and hit_count is not None:
        # Show the MC results from get_forecast
        if avg_hit_day:
            st.success(
                f"🎯 Monte Carlo Simulation indicates an average hit time of **{avg_hit_day:.2f} Months** "
                f"with a probability of **{hit_count/10000:.2%}**."
            )
        else:
            st.warning(
                "⚠️ Target price may not be reached within the given simulation period (Monte Carlo Simulation)."
            )

        # Plot existing MC
        st.subheader("📉 Monte Carlo Simulation Paths")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        paths_arr = np.array(paths)
        ax3.plot(range(1, simulation_days + 1), paths_arr[:100].T, color='gray', alpha=0.1)
        ax3.axhline(y=target_price, color="green", linestyle="--",
                    label=f"Target Price: ${target_price}")
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Price")
        ax3.set_title("Monte Carlo Simulated Stock Price Paths")
        ax3.set_ylim(0, 2000)
        ax3.legend()
        st.pyplot(fig3)


    # Now, plot the daily Bollinger, RSI, MACD *without* SARIMA in the triple subplot
    start_date = "2023-01-01"   
    end_date = date.today().strftime("%Y-%m-%d")
    data_daily = get_daily_stock_data(ticker, start_date, end_date)

    # Debug: See how your columns look
    # st.write("Data Daily Columns:", data_daily.columns)
    # st.write("Data Daily Shape:", data_daily.shape)

    if data_daily.empty:
        logger.error("No daily data found for the given ticker/date range.")
        st.warning("No daily data available for technical indicators.")
    else:
        # 1) Flatten columns if they are multi-level (e.g. outer='Ticker', inner='AAPL')
        if isinstance(data_daily.columns, pd.MultiIndex):
            data_daily.columns = data_daily.columns.droplevel(0)

        # 2) If the remaining column is named after your ticker (e.g. 'AAPL'), rename it to 'Close'
        if ticker in data_daily.columns:
            data_daily.rename(columns={ticker: 'Close'}, inplace=True)

        # Make sure we actually have a 'Close' column
        if 'Close' not in data_daily.columns:
            st.error("No 'Close' column found after flattening/renaming.")
        else:
            # 3) Extract 'Close' as a 1D Series
            close_series = data_daily['Close']

            # If close_series is still a DataFrame with shape (N,1), convert to Series
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]  # take the first column

            close_series = close_series.squeeze()  # remove any extra dimension

            # --- Compute technical indicators on this 1D series ---
            rsi = compute_rsi(close_series)
            sma_20, upper_band, lower_band = compute_bollinger_bands(close_series, window=20)
            macd_line, signal_line, hist = compute_macd(close_series)

            # 4) Squeeze everything to ensure they are 1D
            #    (in case rolling(...) returned a DataFrame with shape (N,1))
            for arr_name in [sma_20, upper_band, lower_band, macd_line, signal_line, hist]:
                if isinstance(arr_name, pd.DataFrame):
                    # If any array is still DataFrame, convert to Series
                    arr_name = arr_name.iloc[:, 0]
                arr_name = arr_name.squeeze()

            # If you need them as local variables again, do:
            sma_20 = sma_20.squeeze()
            upper_band = upper_band.squeeze()
            lower_band = lower_band.squeeze()
            macd_line = macd_line.squeeze()
            signal_line = signal_line.squeeze()
            hist = hist.squeeze()

            # Optional debugging prints
            # st.write("Shapes:", 
            #          "close_series =", close_series.shape,
            #          "sma_20 =", sma_20.shape,
            #          "upper_band =", upper_band.shape,
            #          "lower_band =", lower_band.shape,
            #          "macd_line =", macd_line.shape,
            #          "signal_line =", signal_line.shape,
            #          "hist =", hist.shape)
            
            st.subheader("📊 Technical Indicators")
            # --- Plot subplots ---
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

            # Top: Price + Bollinger
            axes[0].plot(close_series.index, close_series, label='Close Price', color='blue')
            axes[0].plot(sma_20.index, sma_20, label='20-day SMA', color='orange')
            axes[0].fill_between(
                upper_band.index, upper_band, lower_band,
                color='gray', alpha=0.3, label='Bollinger Bands'
            )
            axes[0].legend()
            axes[0].set_title("Close Price + Bollinger Bands")

            # Middle: RSI
            axes[1].plot(close_series.index, rsi, label='RSI', color='purple')
            axes[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
            axes[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].set_title("Relative Strength Index (RSI)")

            # Bottom: MACD
            axes[2].plot(close_series.index, macd_line, label='MACD', color='blue')
            axes[2].plot(close_series.index, signal_line, label='Signal', color='red')
            axes[2].bar(close_series.index, hist, label='Histogram', color='gray', alpha=0.5)
            axes[2].legend()
            axes[2].set_title("MACD (Moving Average Convergence Divergence)")

            plt.tight_layout()
            # Save the figure to session state
            st.session_state.bollinger_fig = fig
            st.pyplot(fig)

            with st.spinner("Analyzing the Technical Indicators Chart"): 
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

                        st.sidebar.subheader("Technical Indicators Chart Insights")
                        st.sidebar.write(description)

                except Exception as e:
                    st.sidebar.error(f"Error in generating insights: {e}")