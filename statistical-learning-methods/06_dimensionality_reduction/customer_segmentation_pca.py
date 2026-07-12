import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. Data Loading and Descriptive Analysis ---
df = pd.read_csv('segmentation_data.csv', index_col=0)
# df.describe() and df.info() help understand the distribution and data types

# --- 2. Exploratory Data Analysis (EDA) ---
# Correlation Heatmap to identify redundant features
corr_matrix = df.corr()
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu')
plt.title('Correlation Heatmap')
plt.show()

# --- 3. Standardization ---
# K-means requires all features to have equal weight (mean=0, std=1)
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
df_std = pd.DataFrame(df_std, columns=df.columns)

# --- 4. Initial K-means (The Elbow Method) ---
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('K-means Clustering (Elbow Method)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# --- 5. Principal Component Analysis (PCA) ---
# Reduce dimensions while retaining maximum variance
pca = PCA()
pca.fit(df_std)

# Create a heatmap of the loadings (PC vs Features)
# Note: df_pca_comp is derived from pca.components_
df_pca_comp = pd.DataFrame(data=pca.components_[:3],
                           columns=df.columns.values,
                           index=['PC 1', 'PC 2', 'PC 3'])

plt.figure(figsize=(12, 9))
sns.heatmap(df_pca_comp, annot=True, vmin=-1, vmax=1, cmap='RdBu')
plt.title('PCs vs Original Features')
plt.show()

# Transform original data into PCA scores
scores_pca = pca.transform(df_std)

# --- 6. K-means + PCA (Improved Clustering) ---
# Run K-means on the transformed PCA scores
kmeans_pca = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

# Create a final dataframe for analysis
df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True),
                                pd.DataFrame(scores_pca[:, :3])], axis=1)
df_segm_pca_kmeans.columns.values[-3:] = ['PC 1', 'PC 2', 'PC 3']
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

# --- 7. Final Labeling and Visualization ---
segment_map = {0: 'standard',
               1: 'career focused',
               2: 'fewer opportunities',
               3: 'well-off'}
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map(segment_map)

# Plotting the clusters by PC 2 and PC 1
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df_segm_pca_kmeans['PC 2'],
                y=df_segm_pca_kmeans['PC 1'],
                hue=df_segm_pca_kmeans['Legend'],
                palette='rainbow')
plt.title('Clusters by PCA Components')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Data Preparation ---
df = pd.read_csv('segmentation_data.csv', index_col=0)
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
pca = PCA()
pca.fit(df_std)

# Define the PCA loadings dataframe (first 3 components)
df_pca_comp = pd.DataFrame(
    data=pca.components_[:3],
    columns=df.columns.values,
    index=['PC 1', 'PC 2', 'PC 3']
)

# --- Combined Visualization with ENHANCED Label Sizes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

# 1. Plot Raw Feature Correlation Matrix
sns.heatmap(df.corr(),
            annot=True,
            fmt=".2f",
            cmap='RdBu',
            ax=ax1,
            cbar=True,
            annot_kws={"size": 14}) # Larger numbers inside cells
ax1.set_title('Correlation Heatmap (Raw Features)', fontsize=22, pad=20)
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=14, rotation=45)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)

# 2. Plot PCA Component Loadings
sns.heatmap(df_pca_comp,
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            cmap='RdBu',
            ax=ax2,
            cbar=True,
            annot_kws={"size": 14}) # Larger numbers inside cells
ax2.set_title('PCs vs Original Features (Loadings)', fontsize=22, pad=20)
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=14, rotation=45)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()
