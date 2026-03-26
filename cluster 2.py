import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Credit Card Customer Data.csv")
df = df.drop(["Sl_No", "Customer Key"], axis=1)

num_data = df.select_dtypes(include="number")
X = StandardScaler().fit_transform(num_data)

best_score = -1
best_k = None
best_labels = None

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    print(f"k={k}, Silhouette Score={score:}")
    
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

print(f"Best number of clusters = {best_k}, Best Silhouette Score = {best_score:}")
df["cluster"] = best_labels

X_pca = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["cluster"], palette="tab10", s=60)
plt.title(f"KMeans Clusters (k={best_k})")
plt.show()

sns.pairplot(df[num_data.columns.tolist() + ["cluster"]], hue="cluster", palette="tab10")
plt.suptitle("Pairplot of Numerical Features by Cluster", y=1.02)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="cluster", data=df, palette="tab10")
plt.title("Number of Customers in Each Cluster")
plt.show()

if "Credit_Limit" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="cluster", y="Credit_Limit", data=df, palette="tab10")
    plt.title("Credit Limit Distribution by Cluster")
    plt.show()

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
kmeans_final.fit(X)
cluster_centers = pd.DataFrame(kmeans_final.cluster_centers_, columns=num_data.columns)
plt.figure(figsize=(10,6))
sns.heatmap(cluster_centers, annot=True, cmap="coolwarm")
plt.title("Cluster Centers Heatmap")
plt.show()

print(df["cluster"].value_counts())
df.to_csv("credit_card_clusters.csv", index=False)







