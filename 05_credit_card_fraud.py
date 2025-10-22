import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 800
df = pd.DataFrame({
    "TransactionAmount": np.random.uniform(1, 20000, n),
    "TransactionHour": np.random.randint(0, 24, n),
    "CustomerAge": np.random.randint(18, 80, n),
    "OnlinePurchase": np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
})
df["fraud_prob"] = (
    (df["TransactionAmount"] / 20000) * 0.5 +
    (df["TransactionHour"] < 5) * 0.3 +
    (df["OnlinePurchase"] == "Yes") * 0.2
)
df["IsFraud"] = np.random.binomial(1, df["fraud_prob"].clip(0, 1)).astype(int)
df = df.drop(columns=["fraud_prob"])

# encoding & scaling
le = LabelEncoder()
df["OnlinePurchase_enc"] = le.fit_transform(df["OnlinePurchase"])
X = df[["TransactionAmount", "TransactionHour", "CustomerAge", "OnlinePurchase_enc"]]
y = df["IsFraud"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("--- Credit Card Fraud Detection ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Reds")
plt.title("Fraud Confusion Matrix"); plt.show()
