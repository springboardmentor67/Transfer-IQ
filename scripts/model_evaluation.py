# =============================================================================
# TransferIQ: Model Evaluation Script
# =============================================================================
# This script calculates regression evaluation metrics to measure how well
# the model predicts football player transfer market values.
# 
# Assumes `y_test` (actual values) and `y_pred` (predicted values) already exist.
# =============================================================================

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# Example Placeholder Data (replace with your actual model output)
# -----------------------------------------------------------------------------
# In your full pipeline, y_test and y_pred come from your trained model:
#   y_pred = model.predict(X_test)
# Uncomment the lines below only if you want to test this script standalone.

# import numpy as np
# np.random.seed(42)
# y_test = np.random.uniform(1e6, 50e6, 100)   # actual market values
# y_pred = y_test + np.random.normal(0, 3e6, 100)  # simulated predictions

# -----------------------------------------------------------------------------
# 1. RMSE — Root Mean Squared Error
# -----------------------------------------------------------------------------
# RMSE measures the average magnitude of errors in the same unit as the target.
# Lower RMSE = better model performance.
# Squaring penalizes large errors more than small ones.

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# -----------------------------------------------------------------------------
# 2. MAE — Mean Absolute Error
# -----------------------------------------------------------------------------
# MAE measures the average absolute difference between actual and predicted values.
# It is easier to interpret: "on average, predictions are off by this amount."
# Less sensitive to large outliers compared to RMSE.

mae = mean_absolute_error(y_test, y_pred)

# -----------------------------------------------------------------------------
# 3. R² Score — Coefficient of Determination
# -----------------------------------------------------------------------------
# R² indicates how much variance in the target variable the model explains.
# Range: (-∞, 1.0] — closer to 1.0 means the model fits the data better.
# An R² of 0 means the model is no better than predicting the mean.

r2 = r2_score(y_test, y_pred)

# -----------------------------------------------------------------------------
# Print Results
# -----------------------------------------------------------------------------
print("=" * 50)
print("  TransferIQ — Model Evaluation Results")
print("=" * 50)
print(f"  RMSE  (Root Mean Squared Error) : €{rmse:,.2f}")
print(f"  MAE   (Mean Absolute Error)     : €{mae:,.2f}")
print(f"  R²    (R-Squared Score)         : {r2:.4f}")
print("=" * 50)

# Interpretation hints
print("\nInterpretation:")
if r2 >= 0.9:
    print("  R² >= 0.90  → Excellent model fit")
elif r2 >= 0.75:
    print("  R² >= 0.75  → Good model fit")
elif r2 >= 0.5:
    print("  R² >= 0.50  → Moderate model fit")
else:
    print("  R² < 0.50   → Poor model fit — consider feature engineering or tuning")

print(f"\n  On average, transfer value predictions are off by €{mae:,.2f}")
print(f"  RMSE penalises large errors more heavily: €{rmse:,.2f}")
