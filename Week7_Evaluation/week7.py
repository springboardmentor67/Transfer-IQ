import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

############################################
# LOAD DATA
############################################

df = pd.read_csv("final_ensemble_dataset.csv")

############################################
# CLEAN DATA (IMPORTANT FIX)
############################################

df = df.replace([np.inf,-np.inf],np.nan)

df = df.fillna(0)

############################################
# DEFINE TARGET + PREDICTIONS
############################################

y_true = df["actual_value"].astype(float)

uni = df["uni_lstm_pred"]

multi = df["multi_lstm_pred"]

enc = df["encoder_lstm_pred"]

############################################
# EVALUATION FUNCTION
############################################

def evaluate(y_true,y_pred,name):

    mae = mean_absolute_error(y_true,y_pred)

    rmse = np.sqrt(mean_squared_error(y_true,y_pred))

    r2 = r2_score(y_true,y_pred)

    print("\n",name)

    print("MAE:",mae)

    print("RMSE:",rmse)

    print("R2:",r2)

    return mae,rmse,r2

############################################
# VALIDATION SPLIT
############################################

X = df[[

"uni_lstm_pred",
"multi_lstm_pred",
"encoder_lstm_pred"

]]

y = df["actual_value"].astype(float)

X_train,X_val,y_train,y_val = train_test_split(

X,
y,

test_size=0.2,
random_state=42

)

############################################
# XGBOOST HYPERPARAMETER TUNING
############################################

print("\nTuning XGBoost...")

params = {

'max_depth':[3,5,7],
'learning_rate':[0.01,0.1],
'n_estimators':[100,200],
'subsample':[0.8,1]

}

model = XGBRegressor(

random_state=42,
n_jobs=-1

)

grid = GridSearchCV(

model,
params,

cv=3,

scoring='neg_mean_squared_error',

verbose=1,

error_score='raise'

)

grid.fit(X_train,y_train)

print("\nBest Parameters:")

print(grid.best_params_)

############################################
# VALIDATION PERFORMANCE
############################################

best_model = grid.best_estimator_

val_pred = best_model.predict(X_val)

evaluate(

y_val,
val_pred,

"Tuned XGBoost Validation"

)

############################################
# FINAL XGB PREDICTIONS
############################################

final_xgb = best_model.predict(X)

############################################
# ENSEMBLE MODEL
############################################

ensemble = (

multi*0.4 +
enc*0.3 +
final_xgb*0.3

)

############################################
# MODEL EVALUATION
############################################

uni_mae,uni_rmse,uni_r2 = evaluate(

y_true,
uni,

"Univariate LSTM"

)

multi_mae,multi_rmse,multi_r2 = evaluate(

y_true,
multi,

"Multivariate LSTM"

)

enc_mae,enc_rmse,enc_r2 = evaluate(

y_true,
enc,

"Encoder LSTM"

)

xgb_mae,xgb_rmse,xgb_r2 = evaluate(

y_true,
final_xgb,

"Tuned XGBoost"

)

ens_mae,ens_rmse,ens_r2 = evaluate(

y_true,
ensemble,

"Ensemble"

)

############################################
# MODEL COMPARISON TABLE
############################################

models = [

"Uni LSTM",
"Multi LSTM",
"Encoder",
"Tuned XGBoost",
"Ensemble"

]

rmse = [

uni_rmse,
multi_rmse,
enc_rmse,
xgb_rmse,
ens_rmse

]

mae = [

uni_mae,
multi_mae,
enc_mae,
xgb_mae,
ens_mae

]

r2 = [

uni_r2,
multi_r2,
enc_r2,
xgb_r2,
ens_r2

]

results = pd.DataFrame({

"Model":models,
"MAE":mae,
"RMSE":rmse,
"R2":r2

})

results = results.sort_values("RMSE")

print("\nMODEL COMPARISON")

print(results)

results.to_csv("model_comparison.csv",index=False)

############################################
# RMSE GRAPH
############################################

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["RMSE"])

plt.title("Model RMSE Comparison")

plt.ylabel("RMSE")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

############################################
# MAE GRAPH
############################################

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["MAE"])

plt.title("Model MAE Comparison")

plt.ylabel("MAE")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

############################################
# R2 GRAPH
############################################

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["R2"])

plt.title("Model R2 Comparison")

plt.ylabel("R2 Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

############################################
# ACTUAL VS PREDICTED
############################################

plt.figure(figsize=(10,5))

plt.plot(
y_true.values,
label="Actual"
)

plt.plot(
final_xgb,
label="Tuned XGBoost"
)

plt.plot(
ensemble,
label="Ensemble"
)

plt.title("Actual vs Predicted")

plt.xlabel("Samples")

plt.ylabel("Transfer Value")

plt.legend()

plt.tight_layout()

plt.show()

############################################
# ERROR DISTRIBUTION
############################################

errors = y_true - ensemble

plt.figure(figsize=(8,5))

plt.hist(errors,bins=30)

plt.title("Prediction Error Distribution")

plt.xlabel("Error")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()

############################################
# SAVE FINAL RESULTS
############################################

df["tuned_xgb_pred"] = final_xgb

df["final_prediction"] = ensemble

df.to_csv(

"final_predictions.csv",

index=False

)

print("\nWeek 7 Completed Successfully")
