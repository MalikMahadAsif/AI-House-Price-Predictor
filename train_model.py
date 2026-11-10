
# Step 1: Import the tools we need
from pathlib import Path
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load the housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Step 3: Separate inputs and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create and train our ML model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)
print("Training completed!")

# Step 6: Predict values for unseen test data
predictions = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: ${mae * 100000:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Step 8: Save the trained model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "house_price_model.pkl")

print("Model saved successfully!")
