from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import datetime as dt
import time as tm
from flask_cors import CORS

# SETTINGS
N_STEPS = 7
LOOKUP_STEPS = [1, 2, 3]
STOCK = 'AAPL'
scaler = MinMaxScaler()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Load saved model
model_path = 'saved_modelv4.h5'  # Adjust path accordingly
model = tf.keras.models.load_model(model_path)

# Data preparation function
def prepare_data(stock, days):
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')
    init_df = yf.get_data(stock, start_date=date_3_years_back, end_date=date_now, interval='1d')
    historical_data = init_df[['close']].reset_index().rename(columns={'index': 'date', 'close': 'price'}).to_dict(orient='records')
    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['scaled_close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

    sequences = deque(maxlen=N_STEPS)
    for entry in init_df[['scaled_close']].values[:-days]:  # Shift window by 'days'
        sequences.append(entry)

    last_sequence = list([s[:len(['scaled_close'])] for s in sequences])
    last_sequence = np.array(last_sequence).astype(np.float32)
    last_sequence = np.expand_dims(last_sequence[-N_STEPS:], axis=0)

    prediction = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return round(float(predicted_price), 2), historical_data

# Prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol')
    predictions, historical_data = zip(*[prepare_data(symbol, step) for step in LOOKUP_STEPS])

    return jsonify({
        "prediction": predictions,
        "historical": historical_data[0]  # Since historical data will be the same for all LOOKUP_STEPS
    })

# Test
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Hello World!"
    })

if __name__ == "__main__":
    app.run()
