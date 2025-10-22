import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 400
df = pd.DataFrame({
    "Mileage": np.random.randint(1000, 200000, n),
    "EngineSize": np.round(np.random.uniform(1.0, 5.0, n), 1),
    "Age": np.random.randint(0, 20, n),
    "FuelType": np.random.choice(["Diesel", "Petrol", "Hybrid"], n)
})
# price formula
df["Price"] = 50000 - 0.05 * df["Mileage"] + 4000 * df["EngineSize"] - 1000 * df["Age"] + np.random.normal(0, 3000, n)

le = LabelEncoder()
df["FuelType_enc"] = le.fit_transform(df["FuelType"])
X = df[["Mileage", "EngineSize", "Age", "FuelType_enc"]]
y = df["Price"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Car Price Estimation ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 :", round(r2_score(y_test, y_pred), 3))

# feature importance
importances = model.feature_importances_
features = X.columns
fi = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print("Top features:", fi)

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
plt.title("Car Price - Actual vs Pred")
plt.grid(alpha=0.3); plt.show()
