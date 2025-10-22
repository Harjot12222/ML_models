import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 🧩 Synthetic Dataset Creation
# ===============================
np.random.seed(42)
n = 300
cities = ["Seattle", "Austin", "Denver", "Boston", "Chicago"]

df = pd.DataFrame({
    "city": np.random.choice(cities, n),
    "sqft_lot": np.random.randint(500, 8000, n),
})

# Create price based on city and lot size (simulate real trends)
city_influence = {
    "Seattle": 550,
    "Austin": 420,
    "Denver": 480,
    "Boston": 600,
    "Chicago": 350
}
df["price"] = df.apply(
    lambda row: (
        row["sqft_lot"] * (city_influence[row["city"]] / 1000)
        + np.random.normal(50000, 10000)
    ),
    axis=1
)

print("\n✅ Synthetic dataset created:")
print(df.head())

# ===============================
# 🧹 Data Cleaning
# ===============================
df = df.dropna()
df = df.rename(columns=lambda x: x.strip())

# ===============================
# 🔢 Feature Encoding & Scaling
# ===============================
le = LabelEncoder()
df["city"] = le.fit_transform(df["city"])

X = df[["city", "sqft_lot"]]
y = df["price"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 🧠 Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# 🏗️ Model Training
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 📈 Model Evaluation
# ===============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ===============================
# 🔮 Sample Predictions
# ===============================
sample = pd.DataFrame({
    "city": ["Seattle", "Austin", "Denver"],
    "sqft_lot": [500, 300, 450]
})
sample["city"] = le.transform(sample["city"])
sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)
for i in range(len(sample)):
    print(f"🏡 {cities[i]} | Lot: {sample['sqft_lot'][i]} sqft → Predicted Price: ${pred[i]:,.2f}")

# ===============================
# 📊 Visualization
# ===============================
errors = np.abs(y_test - y_pred)
plt.scatter(y_test, y_pred, c=errors, cmap="coolwarm", alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.colorbar(label="Absolute Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
