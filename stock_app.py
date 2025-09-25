import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Function to fetch historical stock data
def fetch_stock_data(stock_name, period='1y'):
    try:
        stock = yf.Ticker(stock_name)
        hist = stock.history(period=period, timeout=10)  # Add a timeout
        return hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Function to preprocess data for prediction
def preprocess_data_for_prediction(data, scaler, lookback=60):
    try:
        # Extract the 'Close' prices
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = scaler.transform(close_prices)
        
        # Prepare the input features (X)
        X = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
        
        X = np.array(X)
        
        # Reshape X to 3D for LSTM input
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

# Function to predict stock price
def predict_stock_price(model, data, scaler):
    try:
        # Get the last 60 days of data
        last_60_days = data['Close'].values[-60:].reshape(-1, 1)
        
        # Scale the data
        last_60_days_scaled = scaler.transform(last_60_days)
        
        # Reshape the data for LSTM input
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict the stock price
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        return pred_price[0][0]
    except Exception as e:
        st.error(f"Error predicting stock price: {e}")
        return None

# Streamlit app
def main():
    st.title('Indian Stock Price Prediction App')
    st.write("This app predicts the stock price for Indian stocks using a pre-trained LSTM model.")

    # User inputs
    st.sidebar.header("User Input")
    stock_name = st.sidebar.text_input('Enter Stock Ticker (e.g., RELIANCE.NS or RELIANCE.BO):', 'RELIANCE.NS')
    week_high = st.sidebar.number_input('Enter 52 Week High:', value=150.0)
    week_low = st.sidebar.number_input('Enter 52 Week Low:', value=100.0)

    if st.sidebar.button('Predict'):
        if stock_name:
            # Fetch historical stock data
            st.write(f"Fetching data for {stock_name}...")
            data = fetch_stock_data(stock_name, period='1y')  # Use 1 year of data for prediction
            
            if not data.empty:
                st.write("### Stock Data")
                st.write(data.tail())  # Display the last few rows of the data

                # Load the pre-trained model
                try:
                    model = load_model('stock_prediction_model.h5')
                    st.write("Debug: Model loaded successfully")
                    
                    # Load the scaler used during training
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaler.fit(data['Close'].values.reshape(-1, 1))  # Fit the scaler on the fetched data
                    
                    # Predict stock price
                    st.write("Predicting stock price...")
                    predicted_price = predict_stock_price(model, data, scaler)
                    if predicted_price is not None:
                        st.success(f'Predicted Stock Price for {stock_name}: ₹{predicted_price:.2f}')
                    else:
                        st.error("Prediction failed. Please check the logs.")
                except Exception as e:
                    st.error(f"Error loading the model: {e}. Please train a model first.")
            else:
                st.error('No data found for the given stock ticker. Please check the ticker symbol.')
        else:
            st.error('Please enter a valid stock ticker.')

# Run the Streamlit app
if __name__ == '__main__':
    main()