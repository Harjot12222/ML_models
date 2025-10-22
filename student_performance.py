import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 🧩 Synthetic Dataset Creation
# ===============================
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "test_preparation_course": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),  # 0: No, 1: Yes
    "writing_score": np.random.randint(40, 100, size=n),
})

# Simulate total_score with slight random noise
df["total_score"] = (
    0.5 * df["writing_score"] +
    15 * df["test_preparation_course"] +
    np.random.normal(0, 5, size=n)
)

print("\n✅ Synthetic dataset created:")
print(df.head())

# ===============================
# 🧹 Data Cleaning
# ===============================
df = df.dropna()
df = df.rename(columns=lambda x: x.strip())  # remove accidental spaces

print("\n--- Statistical Summary ---")
print(df.describe())

# ===============================
# 📊 Exploratory Visualization
# ===============================
plt.scatter(df["writing_score"], df["total_score"], color="blue", alpha=0.6)
plt.title("Writing Score vs Total Score")
plt.xlabel("Writing Score")
plt.ylabel("Total Exam Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===============================
# 🔢 Feature Selection
# ===============================
X = df[["test_preparation_course", "writing_score"]]
y = df["total_score"]

# ===============================
# 🧠 Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
# 📊 Results Visualization
# ===============================
plt.scatter(y_test, y_pred, color="purple", alpha=0.6)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Student Scores")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
