import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
positive = ["good", "nice", "love", "excellent", "great", "recommend"]
negative = ["bad", "terrible", "hate", "poor", "disappoint", "refund"]
neutral = ["okay", "average", "fine", "satisfactory"]

reviews = []
labels = []
for _ in range(300):
    if np.random.rand() < 0.45:
        words = np.random.choice(positive, np.random.randint(1, 4))
        reviews.append(" ".join(words))
        labels.append(1)
    elif np.random.rand() < 0.9:
        words = np.random.choice(negative, np.random.randint(1, 4))
        reviews.append(" ".join(words))
        labels.append(0)
    else:
        words = np.random.choice(neutral, np.random.randint(1, 4))
        reviews.append(" ".join(words))
        labels.append(1 if np.random.rand() > 0.5 else 0)

df = pd.DataFrame({"review": reviews, "label": labels})
cv = CountVectorizer()
X = cv.fit_transform(df["review"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Product Review Sentiment ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Sentiment"); plt.show()
