# src/lstm_models.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


# ----------------------------
# Univariate LSTM
# ----------------------------

def build_univariate_lstm(seq_length):

    model = Sequential()

    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(seq_length,1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model



# ----------------------------
# Multivariate LSTM
# ----------------------------

def build_multivariate_lstm(seq_length, n_features):

    model = Sequential()

    model.add(
        LSTM(
            64,
            activation="relu",
            input_shape=(seq_length,n_features)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model



# ----------------------------
# Encoder Decoder LSTM
# ----------------------------

def build_encoder_decoder(seq_length, n_features, future_steps):

    model = Sequential()

    model.add(
        LSTM(
            64,
            activation="relu",
            input_shape=(seq_length,n_features)
        )
    )

    model.add(RepeatVector(future_steps))

    model.add(
        LSTM(
            64,
            activation="relu",
            return_sequences=True
        )
    )

    model.add(TimeDistributed(Dense(1)))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model