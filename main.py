import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Function to get historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to prepare data for LSTM model
def prepare_lstm_data(data, target_col='Close', lookback=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[target_col] = scaler.fit_transform(data[target_col].values.reshape(-1, 1))

    x, y = [], []
    for i in range(len(data) - lookback):
        x.append(data[target_col].values[i:(i + lookback)])
        y.append(data[target_col].values[i + lookback])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], lookback, 1))

    return x, y, scaler

# Function to build and train LSTM model
def build_lstm_model(lookback):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot actual vs. predicted prices
def plot_predictions(actual, predicted, dates):
    plt.plot(dates, actual, label='Actual Prices')
    plt.plot(dates, predicted, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Stock symbol and date range
stock_symbol = 'MSFT'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Get historical stock data
stock_data = get_stock_data(stock_symbol, start_date, end_date)
print(stock_data)

# Prepare data for LSTM model
lookback = 20
x, y, scaler = prepare_lstm_data(stock_data, lookback=lookback)

# Split the data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train LSTM model
model = build_lstm_model(lookback)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[reduce_lr])

# Make predictions on the test set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform the original data for plotting
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs. predicted prices
plot_predictions(actual_prices, predictions, stock_data.index[train_size + lookback:])