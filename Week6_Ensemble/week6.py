import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

##################################
# LOAD DATA
##################################

og=pd.read_csv("player_transfer_value_with_sentiment.csv")

lstm=pd.read_csv("lstm_predictions.csv")

##################################
# MERGE LSTM PREDICTIONS
##################################

df=og.merge(

lstm,

on=["player_name","season"],

how="left"

)

# keep only rows with predictions
df=df.dropna(subset=[
"uni_lstm_pred",
"multi_lstm_pred",
"encoder_lstm_pred"
])

print("Dataset shape:",df.shape)

##################################
# PREPARE TRAINING DATA
##################################

drop_cols=[

"player_name",
"team",
"position",
"market_value_source",
"career_stage",
"most_common_injury",
"sentiment_label",
"season"

]

# create training copy (IMPORTANT)
train_df=df.drop(columns=drop_cols,errors='ignore')

# remove remaining NaN
train_df=train_df.dropna()

##################################
# DEFINE TARGET
##################################

target="market_value_eur"

X=train_df.drop(columns=[target])

y=train_df[target]

##################################
# TRAIN TEST SPLIT
##################################

X_train,X_test,y_train,y_test=train_test_split(

X,
y,

test_size=0.2,

random_state=42

)

##################################
# XGBOOST MODEL
##################################

model=XGBRegressor(

n_estimators=300,

learning_rate=0.05,

max_depth=6,

subsample=0.8,

colsample_bytree=0.8,

random_state=42

)

model.fit(

X_train,
y_train

)

##################################
# PREDICTIONS
##################################

pred=model.predict(X_test)

##################################
# METRICS
##################################

rmse=np.sqrt(

mean_squared_error(

y_test,
pred

))

mae=mean_absolute_error(

y_test,
pred

)

r2=r2_score(

y_test,
pred

)

print("RMSE:",rmse)

print("MAE:",mae)

print("R2:",r2)

##################################
# FEATURE IMPORTANCE
##################################

importance=model.feature_importances_

cols=X.columns

plt.figure(figsize=(8,6))

plt.barh(cols,importance)

plt.title("Feature Importance")

plt.show()

##################################
# ACTUAL VS PREDICTED
##################################

plt.scatter(

y_test,
pred

)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.title("Actual vs Predicted")

plt.show()

##################################
# APPLY MODEL TO FULL DATASET
##################################

full = pd.read_csv("final_dataset_with_lstm.csv")

# remove same non-feature columns
full_model = full.drop(columns=drop_cols,errors='ignore')

# keep same feature structure as training
X_full = full_model[X.columns]

# fill any missing values
X_full = X_full.fillna(0)

full["final_prediction"] = model.predict(X_full)

##################################
# SAVE FINAL
##################################

full.to_csv(

"final_ensemble_dataset.csv",

index=False

)

print("Final dataset shape:",full.shape)

print("Players:",full["player_name"].nunique())
