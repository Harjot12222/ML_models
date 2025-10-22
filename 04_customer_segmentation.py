import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

np.random.seed(42)
n = 400
df = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "AnnualIncome": np.random.randint(15000, 120000, n),
    "SpendingScore": np.random.randint(1, 100, n),
})

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["Segment"] = labels

pca = PCA(n_components=2)
proj = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette="tab10", legend="full", s=60)
plt.title("Customer Segmentation (PCA projection)")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.tight_layout(); plt.show()

print("Segment sizes:", df["Segment"].value_counts().to_dict())
# show centroids in original feature scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("Centroids (Age, Income, Score):")
print(np.round(centroids, 2))
