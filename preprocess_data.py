import pandas as pd
from sklearn.model_selection import train_test_split
from data import read_data, convert_data
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from config import VOCAB_SIZE, MAX_WORDS


def preprocess_data():
    df = read_data()
    df = convert_data(df)

    # drop null values
    df = df.dropna()

    max_words = MAX_WORDS
    df = df[df["Text"].apply(len) < max_words]

    # split data into training and testing files
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

    # tokenize the data
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df['Text'])

    max_len = df['Text'].apply(len).max()

    # pad the train and test sequences after fitting on tokenizer
    train_df['Text'] = tokenizer.texts_to_sequences(train_df['Text'])
    X_train = pad_sequences(train_df['Text'],
                            maxlen=max_len,
                            padding='post')
    y_train = train_df['Target']

    test_df['Text'] = tokenizer.texts_to_sequences(test_df['Text'])
    X_test = pad_sequences(test_df['Text'],
                           maxlen=max_len,
                           padding='post')
    y_test = test_df['Target']

    return X_train, y_train, X_test, y_test
