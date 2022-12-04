from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional


def build_model():
    model = Sequential()
