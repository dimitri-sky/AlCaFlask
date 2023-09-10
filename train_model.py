import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

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
    'BA',    # The Boeing Company - Industrial/Aerospace
    'T',     # AT&T Inc. - Telecommunications
    'GE',    # General Electric - Industrial/Conglomerate
    'DIS',   # The Walt Disney Company - Entertainment/Media
    'GILD',  # Gilead Sciences, Inc. - Biotechnology
    'F',     # Ford Motor Company - Automotive
    'AMZN',  # Amazon.com Inc. - E-commerce/Technology
    'NFLX',  # Netflix Inc. - Entertainment/Streaming
    'DAL',   # Delta Air Lines Inc. - Airline
    'GOOGL',  # Alphabet Inc. - Technology/Internet
    'BAC',    # Bank of America - Financial/Banking
    'JNJ',    # Johnson & Johnson - Health Care/Consumer Goods
    'HD',     # The Home Depot - Retail/Home Improvement
    'CVX',    # Chevron Corporation - Energy/Oil
    'PEP',    # PepsiCo, Inc. - Consumer Goods/Food & Beverage
    'VZ',     # Verizon Communications - Telecommunications
    'BA',     # The Boeing Company - Aerospace/Defense
    'BIIB',   # Biogen Inc. - Biotechnology
    'EBAY',   # eBay Inc. - E-commerce/Technology
    'DIS',    # The Walt Disney Company - Entertainment/Media
    'UAL',    # United Airlines Holdings - Airline
]

BATCH_SIZE = 8
EPOCHS = 80

date_now = tm.strftime('%Y-%m-%d')
date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

def PrepareData(STOCK, days):
    try:
        init_df = yf.get_data(STOCK, start_date=date_3_years_back, end_date=date_now, interval='1d')
    except Exception as e:
        print(f"An error occurred while fetching data for {STOCK}: {str(e)}")
        return [], []
    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    scaler = MinMaxScaler()
    init_df['scaled_close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))
    df = init_df.copy()
    df['future'] = df['scaled_close'].shift(-days)
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)

    for entry, target in zip(df['scaled_close'].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            sequence_data.append([np.array(sequences), target])

    # construct the X's and Y's
    X, Y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        Y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Combine data from all stocks
all_X, all_Y = [], []
for STOCK in STOCKS:
    X, Y = PrepareData(STOCK, 3) # 3 days
    all_X.append(X)
    all_Y.append(Y)

all_X = np.concatenate(all_X, axis=0)
all_Y = np.concatenate(all_Y, axis=0)

# Define and train the model
model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, 1)))
model.add(Dropout(0.3))
model.add(LSTM(120, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(all_X, all_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

model.summary()

# Predictions for each stock
predictions = {}
for STOCK in STOCKS:
    X, _ = PrepareData(STOCK, 3)
    X = X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)
    prediction = model.predict(X)
    predictions[STOCK] = prediction

# Print predictions
for STOCK, prediction in predictions.items():
    print(f"{STOCK} prediction for upcoming 3 days: {prediction}")

model_path = "saved_modelv4.h5"
model.save(model_path)
print(f"Model saved to {model_path}")
