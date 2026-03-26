import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("segmentation data.csv")
df = df.drop("ID", axis=1)
num_data = df.select_dtypes(include=np.number)

scaler = StandardScaler()
X = scaler.fit_transform(num_data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

best_eps = None
best_score = -1
best_labels = None

for eps in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    if n_clusters > 1:
        score = silhouette_score(X, labels)
    else:
        score = -1
    
    print(f"eps={eps:.1f} --> clusters={n_clusters}, noise={n_noise}, silhouette={score:.3f}")
    
    if score > best_score:
        best_score = score
        best_eps = eps
        best_labels = labels

print(f"\nBest eps = {best_eps}, Silhouette score = {best_score:.3f}")
df["cluster"] = best_labels

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["cluster"], palette="tab10", s=60)
plt.title(f"DBSCAN Clusters (eps={best_eps})")
plt.show()

print(df["cluster"].value_counts())
df.to_csv("data_with_dbscan_clusters.csv", index=False)



