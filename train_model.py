import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to fetch historical stock data
def fetch_stock_data(stock_name, period='5y'):
    stock = yf.Ticker(stock_name)
    hist = stock.history(period=period)
    return hist

# Function to preprocess data for training
def preprocess_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to train the LSTM model
def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Main function to train and save the model
def train_and_save_model(stock_name):
    # Fetch historical stock data
    print(f"Fetching data for {stock_name}...")
    data = fetch_stock_data(stock_name, period='5y')  # Use 5 years of data for training
    
    if not data.empty:
        # Preprocess the data
        print("Preprocessing data...")
        X_train, y_train, scaler = preprocess_data(data)
        
        # Train the model
        print("Training the model...")
        model = train_model(X_train, y_train, epochs=50, batch_size=32)
        
        # Save the model
        model.save('stock_prediction_model.h5')
        print("Model trained and saved successfully as 'stock_prediction_model.h5'!")
    else:
        print("No data found for the given stock ticker. Please check the ticker symbol.")

# Train the model for a specific stock (e.g., RELIANCE.NS)
train_and_save_model('RELIANCE.NS')