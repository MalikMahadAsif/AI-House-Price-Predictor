
# Import libraries
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 3. Recreate the original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Load our previously trained model
model = joblib.load("models/house_price_model.pkl")

# 5. Make predictions on the test data
predictions = model.predict(X_test)

# Convert prices into US dollars
actual_usd = y_test.to_numpy() * 100000
predicted_usd = predictions * 100000

# 6. Calculate evaluation metrics
mae = mean_absolute_error(actual_usd, predicted_usd)
r2 = r2_score(actual_usd, predicted_usd)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Create a folder for graphs
Path("plots").mkdir(exist_ok=True)

# 7. Graph 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(actual_usd, predicted_usd, alpha=0.25, s=10)
plt.plot(
    [0, 500000], [0, 500000],
    linestyle="--", color="red"
)
plt.xlabel("Actual Housing Value ($)")
plt.ylabel("Predicted Housing Value ($)")
plt.title("Actual vs Predicted Housing Values")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

# 8. Graph 2: Prediction errors
errors = predicted_usd - actual_usd

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50)
plt.axvline(0, linestyle="--", color="red")
plt.xlabel("Prediction Error ($)")
plt.ylabel("Number of Predictions")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("plots/prediction_errors.png")
plt.close()

# 9. Graph 3: Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

print("Evaluation graphs saved in the plots folder!")
