import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 600
df = pd.DataFrame({
    "ApplicantIncome": np.random.randint(2000, 20000, n),
    "CoapplicantIncome": np.random.randint(0, 10000, n),
    "LoanAmount": np.random.randint(50, 700, n),
    "CreditHistory": np.random.choice([0, 1], n, p=[0.15, 0.85]),
    "PropertyArea": np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.4, 0.4, 0.2])
})
df["LoanStatus"] = np.where(
    (df["ApplicantIncome"] > 5000) & (df["CreditHistory"] == 1) & (df["LoanAmount"] < 400),
    1, 0
)

le = LabelEncoder()
df["PropertyArea_enc"] = le.fit_transform(df["PropertyArea"])

X = df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "CreditHistory", "PropertyArea_enc"]]
y = df["LoanStatus"]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("--- Loan Approval Prediction ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Reject", "Approve"]).plot(cmap="viridis")
plt.title("Loan Approval Confusion Matrix"); plt.show()
