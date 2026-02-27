import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def lstm(window=20, units=64, dropout=0.2):
    m = Sequential()
    m.add(LSTM(units, input_shape=(window, None)))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m