import tensorflow as tf
import numpy as np
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import time as tm
from collections import deque

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# SETTINGS
N_STEPS = 7
LOOKUP_STEPS = [1, 2, 3]
STOCKS = [
    'AAPL',  # Apple Inc. - Technology/Electronics
    'JPM',   # JPMorgan Chase & Co. - Financial/Banking
    'PFE',   # Pfizer Inc. - Health Care/Pharmaceuticals
    'WMT',   # Walmart Inc. - Retail
    'XOM',   # Exxon Mobil Corporation - Energy/Oil
    'KO',    # The Coca-Cola Company - Consumer Goods/Beverages
    'T',     # AT&T Inc. - Telecommunications
    'GE',    # General Electric - Industrial/Conglomerate
    'GILD',  # Gilead Sciences, Inc. - Biotechnology
    'AMZN',  # Amazon.com Inc. - E-commerce/Technology
    'NFLX',  # Netflix Inc. - Entertainment/Streaming
    'DAL',   # Delta Air Lines Inc. - Airline
]

# Initialize summary statistics
total_directional_accuracy = 0
total_mae = 0
total_mse = 0
total_rmse = 0

scaler = MinMaxScaler()

# Load saved model
model_path = 'saved_modelv4.h5'
model = tf.keras.models.load_model(model_path)

def test_stock(stock):
    global total_directional_accuracy, total_mae, total_mse, total_rmse  # Declare global variables to update them

    # Data preparation function similar to prepare_data
    def prepare_data(stock, days):
        date_now = tm.strftime('%Y-%m-%d')
        date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')
        init_df = yf.get_data(stock, start_date=date_3_years_back, end_date=date_now, interval='1d')
        init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
        init_df['scaled_close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

        sequences = deque(maxlen=N_STEPS)
        X, Y = [], []
        for entry, target in zip(init_df['scaled_close'].values[:-days], init_df['scaled_close'].values[days:]):
            sequences.append(entry)
            if len(sequences) == N_STEPS:
                X.append(np.array(sequences))
                Y.append(target)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # Prepare data
    X, Y = prepare_data(stock, 1)  # 1-day lookup
    X = np.expand_dims(X, axis=2)  # Reshape for LSTM

    # Make predictions
    predictions = model.predict(X)
    predictions = np.squeeze(scaler.inverse_transform(predictions.reshape(-1, 1)))  # Reshape and descale predictions
    Y_actual = np.squeeze(scaler.inverse_transform(Y.reshape(-1, 1)))  # Reshape and descale actual values

    # Calculate directional accuracy
    correct_count = 0
    total_count = len(Y_actual)

    for i in range(1, total_count):
        actual_direction = Y_actual[i] - Y_actual[i-1]
        predicted_direction = predictions[i] - predictions[i-1]

        if (actual_direction > 0 and predicted_direction > 0) or (actual_direction < 0 and predicted_direction < 0):
            correct_count += 1

    directional_accuracy = (correct_count / (total_count - 1)) * 100

    # Calculate MAE, MSE, RMSE
    mae = mean_absolute_error(Y_actual, predictions)
    mse = mean_squared_error(Y_actual, predictions)
    rmse = math.sqrt(mse)

    # Update summary statistics
    total_directional_accuracy += directional_accuracy
    total_mae += mae
    total_mse += mse
    total_rmse += rmse

    print(f"Results for {stock}:")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")


# Loop over stocks and test each one
for stock in STOCKS:
    test_stock(stock)

# Calculate average summary statistics
num_stocks = len(STOCKS)
avg_directional_accuracy = total_directional_accuracy / num_stocks
avg_mae = total_mae / num_stocks
avg_mse = total_mse / num_stocks
avg_rmse = total_rmse / num_stocks

# Print summary statistics
print("Summary of All Stocks:")
print(f"Average Directional Accuracy: {avg_directional_accuracy:.2f}%")
print(f"Average Mean Absolute Error: {avg_mae:.4f}")
print(f"Average Mean Squared Error: {avg_mse:.4f}")
print(f"Average Root Mean Squared Error: {avg_rmse:.4f}")