import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "Age": np.random.randint(20, 85, n),
    "BMI": np.round(np.random.uniform(16, 40, n), 1),
    "SystolicBP": np.random.randint(90, 180, n),
    "Cholesterol": np.random.randint(150, 300, n)
})
# risk score synthetic formula
df["RiskScore"] = 0.03 * df["Age"] + 0.4 * (df["BMI"]/40) + 0.02 * df["SystolicBP"] + 0.01 * df["Cholesterol"] + np.random.normal(0, 0.5, n)

X = df[["Age", "BMI", "SystolicBP", "Cholesterol"]]
y = df["RiskScore"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Healthcare Risk Scoring ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("R2 :", round(r2_score(y_test, y_pred), 3))

plt.scatter(y_test, y_pred, alpha=0.6); plt.xlabel("Actual Risk"); plt.ylabel("Predicted Risk")
plt.title("Risk Score: Actual vs Predicted"); plt.grid(alpha=0.3); plt.show()
