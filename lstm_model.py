from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional

from config import VOCAB_SIZE


def build_lstm_model(max_len, embedding_len=128, lstm_units=[64, 128, 128]):
    """
    Functionality to build a simple bidirectional lstm model with 64 lstm units.

    :param max_len: maximum input length of the sequences
    :param embedding_len: the dimension of the output of the embedding layer
    :param lstm_units: the number of lstm units required
    :return: the lstm model
    """

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=embedding_len,
                        input_length=max_len))
    for units in lstm_units[:-1]:
        model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(lstm_units[-1]))
    model.add(Dense(1, activation='sigmoid'))
    return model
