import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Fetch monthly stock data
def get_monthly_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print("stock_data :",stock_data)
    # Ensure index is in datetime format
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data.resample('M').last()['Close']

# Define stock symbol and date range
ticker_symbol = 'AAPL'
start_date = "2000-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

button = st.button("Predict")

if button:
    # Get historical data
    stock_prices = get_monthly_stock_data(ticker_symbol, start_date, end_date)

    # Check if data was fetched successfully
    if stock_prices.empty:
        st.error(f"No data found for ticker {ticker_symbol}. Please check the ticker symbol or date range.")
        st.stop()

    # Fit ARIMA model with stationarity/invertibility checks disabled
    arima_model = ARIMA(stock_prices, order=(2, 1, 2), 
                        enforce_stationarity=False, enforce_invertibility=False)
    arima_fitted = arima_model.fit()

    # Fit SARIMA model with stationarity/invertibility checks disabled
    sarima_model = SARIMAX(stock_prices, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), 
                        enforce_stationarity=False, enforce_invertibility=False)
    sarima_fitted = sarima_model.fit()


    # Forecast for the next 18 months
    future_dates = [stock_prices.index[-1] + timedelta(days=30*i) for i in range(1, 19)]
    arima_forecast = arima_fitted.forecast(steps=18)
    sarima_forecast = sarima_fitted.forecast(steps=18)

    # LSTM model
    restored_model = load_model('AAPL_GRU_monthly.h5')

    # Get historical stock data for LSTM
    stock_data = yf.download('AAPL', start=start_date, end=end_date)
    stock_data.index = pd.to_datetime(stock_data.index)  # Ensure datetime index
    stock_data = stock_data.resample('M').last()  # Resample to monthly end dates

    # Extract the last 15 months closing prices
    closing_prices_last_15_months = stock_data['Close'].values[-15:].reshape(-1, 1)

    prediction = []
    if len(closing_prices_last_15_months) >= 15:
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_prices_last_15_months_scaled = scaler.fit_transform(closing_prices_last_15_months)

        # Reshape for LSTM
        input_features = np.reshape(closing_prices_last_15_months_scaled, (1, 1, 15))

        # Make predictions for the next 18 months
        print("Predicted Closing Prices for the Next 18 Months:")
        for i in range(1, 19):
            predicted_scaled_prices = restored_model.predict(input_features)
            predicted_prices = scaler.inverse_transform(predicted_scaled_prices.reshape(-1, 1))
            print(f"Month {i}: ${predicted_prices[0][0]:.2f}")
            prediction.append(predicted_prices[0][0])
            input_features = np.roll(input_features, shift=-1)
            input_features[0, 0, -1] = predicted_scaled_prices[0, 0]

    # Convert to DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'ARIMA_Prediction': arima_forecast,
        'SARIMA_Prediction': sarima_forecast,
        'LSTM_Prediction': prediction
    })
    forecast_df.set_index('Date', inplace=True)

    # Streamlit App
    st.title("Stock Price Prediction using ARIMA, SARIMA, and LSTM")
    st.subheader("Forecasted Stock Prices")
    st.dataframe(forecast_df)

    st.subheader("Stock Price Forecast")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_prices, label='Actual Prices', color='blue')
    ax.plot(forecast_df['ARIMA_Prediction'], label='ARIMA Prediction', linestyle='dashed', color='red')
    ax.plot(forecast_df['SARIMA_Prediction'], label='SARIMA Prediction', linestyle='dashed', color='green')
    ax.plot(forecast_df['LSTM_Prediction'], label='LSTM Prediction', linestyle='dashed', color='orange')
    ax.set_xlabel('Year')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'{ticker_symbol} Stock Price Prediction (ARIMA vs SARIMA vs LSTM)')
    ax.legend()
    st.pyplot(fig)
