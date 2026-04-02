##################################
# IMPORTS
##################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Input,RepeatVector,TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

##################################
# LOAD DATA
##################################

df=pd.read_csv("player_transfer_value_with_sentiment.csv")

df=df.sort_values(["player_name","season"])

SEQ_LENGTH=4

##################################
# SEQUENCE FUNCTION
##################################

def create_sequences(data,seq_length):

    X=[]
    y=[]

    for i in range(len(data)-seq_length):

        X.append(data[i:i+seq_length])

        y.append(data[i+seq_length][0])

    return np.array(X),np.array(y)

##################################
# BUILD DATA (CORRECT PLAYER GROUPING)
##################################

def build_data(features):

    scaler=MinMaxScaler()

    X_train=[]
    X_test=[]

    y_train=[]
    y_test=[]

    test_rows=[]

    for player in df["player_name"].unique():

        player_df=df[
        df["player_name"]==player
        ].copy()

        if len(player_df)<=SEQ_LENGTH+1:
            continue

        scaled=scaler.fit_transform(
        player_df[features]
        )

        X,y=create_sequences(
        scaled,
        SEQ_LENGTH
        )

        if len(X)<2:
            continue

        split=max(1,int(0.8*len(X)))

        X_train.extend(X[:split])

        if split < len(X):

            X_test.extend(X[split:])
            y_test.extend(y[split:])

            rows=player_df.iloc[
            SEQ_LENGTH+split:
            ][["player_name","season","market_value_eur"]]

            test_rows.extend(rows.values)

        y_train.extend(y[:split])

    X_train=np.array(X_train)
    X_test=np.array(X_test)

    y_train=np.array(y_train)
    y_test=np.array(y_test)

    # safety (prevents crashes)
    if len(X_test)==0:

        X_test=X_train[:1]
        y_test=y_train[:1]

        test_rows=test_rows[:1]

    return(

        X_train,
        X_test,

        y_train,
        y_test,

        scaler,

        test_rows

    )

##################################
# FEATURES
##################################

multi_features=[

"market_value_eur",
"goals",
"assists",
"total_days_injured",
"vader_compound_score",
"minutes_played"

]

##################################
# BUILD DATA ONCE
##################################

print("Building sequences")

X_train,X_test,y_train,y_test,multi_scaler,pred_rows = build_data(multi_features)

##################################
# UNIVARIATE MODEL
##################################

print("Training Univariate")

X_train_uni = X_train[:,:,0].reshape(
X_train.shape[0],
SEQ_LENGTH,
1
)

X_test_uni = X_test[:,:,0].reshape(
X_test.shape[0],
SEQ_LENGTH,
1
)

uni_model=Sequential([

Input(shape=(SEQ_LENGTH,1)),

LSTM(32),

Dense(1)

])

uni_model.compile(
optimizer='adam',
loss='mse'
)

history_uni=uni_model.fit(

X_train_uni,
y_train,

epochs=40,
batch_size=8,

validation_data=(X_test_uni,y_test),

callbacks=[EarlyStopping(patience=6)]

)

uni_pred=uni_model.predict(X_test_uni)

##################################
# MULTIVARIATE MODEL
##################################

print("Training Multivariate")

multi_model=Sequential([

Input(shape=(SEQ_LENGTH,len(multi_features))),

LSTM(32),

Dense(1)

])

multi_model.compile(
optimizer='adam',
loss='mse'
)

history_multi=multi_model.fit(

X_train,
y_train,

epochs=40,
batch_size=8,

validation_data=(X_test,y_test),

callbacks=[EarlyStopping(patience=6)]

)

multi_pred=multi_model.predict(X_test)

##################################
# ENCODER DECODER MODEL
##################################

print("Training Encoder")

enc_model=Sequential([

Input(shape=(SEQ_LENGTH,len(multi_features))),

LSTM(64),

RepeatVector(1),

LSTM(32,return_sequences=True),

TimeDistributed(Dense(1))

])

enc_model.compile(
optimizer='adam',
loss='mse'
)

y_train_enc=y_train.reshape(-1,1,1)

y_test_enc=y_test.reshape(-1,1,1)

history_enc=enc_model.fit(

X_train,
y_train_enc,

epochs=40,
batch_size=8,

validation_data=(X_test,y_test_enc),

callbacks=[EarlyStopping(patience=6)]

)

enc_pred=enc_model.predict(X_test)

enc_pred=enc_pred.reshape(-1)

##################################
# INVERSE SCALING
##################################

dummy=np.zeros(
(len(multi_pred),len(multi_features))
)

dummy[:,0]=multi_pred.flatten()

multi_pred=multi_scaler.inverse_transform(dummy)[:,0]

dummy[:,0]=enc_pred

enc_pred=multi_scaler.inverse_transform(dummy)[:,0]

dummy_y=np.zeros(
(len(y_test),len(multi_features))
)

dummy_y[:,0]=y_test

y_test_multi=multi_scaler.inverse_transform(dummy_y)[:,0]

##################################
# METRICS
##################################

rmse=np.sqrt(
mean_squared_error(
y_test_multi,
multi_pred
))

mae=mean_absolute_error(
y_test_multi,
multi_pred
)

print("RMSE:",rmse)

print("MAE:",mae)

##################################
# LOSS CURVE
##################################

plt.plot(
history_multi.history['loss'],
label="train"
)

plt.plot(
history_multi.history['val_loss'],
label="validation"
)

plt.title("Training Loss")

plt.legend()

plt.show()

##################################
# SAVE PREDICTIONS (CORRECT)
##################################

pred_rows=pd.DataFrame(

pred_rows,

columns=[

"player_name",
"season",
"market_value_eur"

]

)

results=pd.DataFrame({

"player_name":pred_rows["player_name"],

"season":pred_rows["season"],

"actual_value":pred_rows["market_value_eur"],

"uni_lstm_pred":uni_pred.flatten(),

"multi_lstm_pred":multi_pred,

"encoder_lstm_pred":enc_pred

})

results.to_csv(

"lstm_predictions.csv",

index=False

)

print("Week 5 completed correctly")
