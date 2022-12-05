from lstm_model import build_lstm_model
from preprocess_data import preprocess_data


def test_model(evaluation_metrics=None):
    """
    Simple test script
    """
    if evaluation_metrics is None:
        evaluation_metrics = ['accuracy']
    X_train, y_train, X_test, y_test = preprocess_data()

    model = build_lstm_model(49)

    if evaluation_metrics is None:
        evaluation_metrics = ['accuracy']

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=evaluation_metrics)
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=10,
                        validation_split=0.2)

    model.evaluate(X_test, y_test)
