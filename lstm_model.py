from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from config import VOCAB_SIZE


def build_lstm_model(max_len, embedding_len=128, lstm_units=64):
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dense(1, activation='sigmoid'))
    return model

print(build_lstm_model(49))