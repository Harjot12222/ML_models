import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def create_datasets():
    np.random.seed(42)
    # classification: fraud
    n = 600
    fraud = pd.DataFrame({
        "Amt": np.random.uniform(1, 20000, n),
        "Hour": np.random.randint(0, 24, n),
        "Online": np.random.choice(["Yes", "No"], n)
    })
    fraud["IsFraud"] = np.random.binomial(1, ((fraud["Amt"]/20000)*0.5 + (fraud["Hour"]<5)*0.3 + (fraud["Online"]=="Yes")*0.2).clip(0,1))

    # regression: house
    n = 400
    house = pd.DataFrame({
        "Area": np.random.randint(500, 4000, n),
        "Rooms": np.random.randint(1, 6, n),
        "Age": np.random.randint(0, 50, n)
    })
    house["Price"] = 50000 + house["Area"]*50 + house["Rooms"]*10000 - house["Age"]*400 + np.random.normal(0, 5000, n)

    return {"fraud": fraud, "house": house}

def preprocess(df, target_col):
    df = df.copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))
    return X_scaled, y

datasets = create_datasets()
print("Available datasets:", list(datasets.keys()))
choice = input("Choose dataset name (fraud/house): ").strip().lower()
if choice not in datasets:
    print("Invalid choice. Defaulting to 'house'."); choice = "house"
df = datasets[choice]
target = "IsFraud" if choice == "fraud" else "Price"
X, y = preprocess(df, target)
is_classification = len(np.unique(y)) == 2 if np.issubdtype(y.dtype, np.integer) else False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)

if is_classification:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix = None
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d"); plt.title("Confusion Matrix"); plt.show()
    except Exception:
        pass
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("--- Regression Metrics ---")
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
    print("R2 :", round(r2_score(y_test, y_pred), 3))
    plt.scatter(y_test, y_pred, alpha=0.6); plt.title("Actual vs Pred"); plt.show()

# sample prediction (simple)
try:
    do_pred = input("Do you want to predict a sample? (y/n): ").strip().lower()
    if do_pred == "y":
        sample_vals = []
        for col in df.drop(columns=[target]).columns:
            val = input(f"Enter value for {col} (press Enter for default): ")
            if val == "":
                sample_vals.append(df[col].iloc[0])
            else:
                sample_vals.append(float(val) if df[col].dtype.kind in "fi" else val)
        # preprocess sample
        sample_df = pd.DataFrame([sample_vals], columns=df.drop(columns=[target]).columns)
        for c in sample_df.select_dtypes(include=["object"]).columns:
            sample_df[c] = LabelEncoder().fit_transform(sample_df[c].astype(str))
        sample_scaled = MinMaxScaler().fit(df.drop(columns=[target])).transform(sample_df)
        pred = model.predict(sample_scaled)
        print("Prediction:", pred)
except Exception as e:
    print("Sample prediction skipped:", e)
