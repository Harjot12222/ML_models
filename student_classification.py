import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ===============================
# 🧩 Synthetic Dataset Creation
# ===============================
np.random.seed(42)
n = 400

df = pd.DataFrame({
    "Study Hours per Week": np.random.uniform(1, 40, n),
    "Attendance Rate": np.random.uniform(50, 100, n),
    "Previous Grades": np.random.uniform(40, 100, n)
})

# Generate pass/fail outcome
df["Passed"] = np.where(
    (df["Study Hours per Week"] > 15) &
    (df["Attendance Rate"] > 70) &
    (df["Previous Grades"] > 60),
    1, 0
)

print("\n✅ Synthetic dataset created:")
print(df.head())

# ===============================
# 🔢 Feature Split & Scaling
# ===============================
X = df[["Study Hours per Week", "Attendance Rate", "Previous Grades"]]
y = df["Passed"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 🧠 Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 🏗️ Model Training
# ===============================
model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# 📈 Model Evaluation
# ===============================
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failed", "Passed"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Student Pass Prediction")
plt.show()

# ===============================
# 🔮 Sample Prediction
# ===============================
sample = np.array([[25, 85, 75]])  # 25 study hours, 85% attendance, 75% previous grade
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)

print("\n🎯 Prediction: Student will",
      "PASS ✅" if pred[0] == 1 else "FAIL ❌")
