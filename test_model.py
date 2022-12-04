from lstm_model import build_lstm_model
from preprocess_data import preprocess_data


def test_model():
    X_train, y_train, X_test, y_test = preprocess_data()

    model = build_lstm_model(49)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=64, epochs=10)

    model.evaluate(X_test, y_test)
