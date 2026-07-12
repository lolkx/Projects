import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Data Generation
X, y = make_blobs(n_samples=500, centers=5, random_state=42, cluster_std=1.2)

# 2. Calculate WCSS for different K values
wcss = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 3. Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='--', color='b')

# Adding labels and title
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_values) # Show all k values on the x-axis
plt.grid(True, linestyle=':', alpha=0.6)

# Highlighting the "Elbow" (In this dataset, it should be at k=5)
plt.annotate('Elbow Point', xy=(5, wcss[4]), xytext=(7, wcss[2]),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12, color='red')

plt.show()


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)
# --- Gráfico 2: K-Means (Agrupamiento) ---
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, edgecolors='k', cmap='tab10')
# Dibujar los centroides
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroides')
plt.title("K-Means: Clustering")
plt.legend()

plt.tight_layout()
plt.show()
