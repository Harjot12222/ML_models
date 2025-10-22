import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.random.seed(42)
n = 730  # two years daily
dates = pd.date_range(start="2023-01-01", periods=n)
promo = np.random.choice([0, 1], n, p=[0.8, 0.2])
sales = 200 + np.sin(np.arange(n) / 50) * 20 + promo * 30 + np.random.normal(0, 10, n)

df = pd.DataFrame({"Date": dates, "Sales": sales, "Promo": promo})
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

X = df[["Month", "DayOfWeek", "IsWeekend", "Promo"]]
y = df["Sales"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
train_size = int(len(df) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Sales Forecasting ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("R2 :", round(r2_score(y_test, y_pred), 3))

plt.figure(figsize=(10, 4))
plt.plot(df["Date"][train_size:], y_test.values, label="Actual")
plt.plot(df["Date"][train_size:], y_pred, label="Predicted")
plt.legend(); plt.title("Sales Forecasting"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
