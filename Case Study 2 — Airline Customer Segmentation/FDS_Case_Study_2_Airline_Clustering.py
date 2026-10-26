# %% [markdown]
# # Case Study 2: Clustering Analysis on East-West Airlines Customer Data
# 
# 
# 
# | Field | Details |
# 
# |---|---|
# 
# | **Subject** | Fundamentals of Data Science |
# 
# | **College** | Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad |
# 
# | **Student** | Katthi Goutham Reddy |
# 
# | **Roll No** | 160124733179 |
# 
# | **Section** | CSE-3 |
# 
# | **Year** | III Year, V Semester (2026-27) |
# 
# | **Faculty** | Dr. S. Durga Devi |
# 
# | **Date** | 26-10-2026 |
# 
# | **GitHub** | https://github.com/katthi-goutham-reddy/FDS-case-study |
# 
# 
# 
# ## Objective
# 
# 
# 
# Apply K-Means, Hierarchical (Agglomerative), and DBSCAN clustering to segment East-West Airlines frequent flyer customers based on their travel and engagement behaviour. Compare the three algorithms and derive actionable marketing insights for each customer segment.

# %% [markdown]
# ## Section 1: Data Preprocessing

# %% [markdown]
# **Purpose:** Load the dataset, understand its structure, check for missing values and duplicates, and display the data dictionary from the Description sheet.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import os
import json

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

df_raw = pd.read_excel('EastWestAirlines.xlsx', sheet_name='data')
df_desc = pd.read_excel('EastWestAirlines.xlsx', sheet_name='Description', header=None)

print('Shape:', df_raw.shape)
print('Columns:', list(df_raw.columns))
df_raw.head()

# %% [markdown]
# **Observation:** The raw East-West Airlines dataset contains 3,999 rows and 12 columns. Columns include customer identifier ('ID#'), frequent flyer account balance ('Balance'), qualification miles, credit card tier miles (cc1, cc2, cc3), bonus miles, bonus transactions, flight miles in 12 months, flight transactions in 12 months, days since enrollment, and award flight status ('Award?').

# %% [markdown]
# **Purpose:** Display the data dictionary, dtypes, missing values, duplicates, and descriptive statistics.

# %%
print('=== Data Dictionary (Description sheet) ===')
print(df_desc.to_string())
print()
print('=== Data Types ===')
print(df_raw.dtypes)
print()
print('=== Missing Values ===')
missing = df_raw.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else 'No missing values')
print()
print('=== Duplicates ===')
print(f'Duplicate rows: {df_raw.duplicated().sum()}')
print()
print('=== Descriptive Statistics ===')
df_raw.describe()

# %% [markdown]
# **Observation:** The dataset has no missing values across all 12 columns. There are 0 duplicate rows. All 12 columns are integer-typed. The Balance column has a wide range from 0 to 1,704,480 miles with mean 73,601 and std 100,776, indicating substantial dispersion.

# %% [markdown]
# **Purpose:** Drop `ID#` (a row identifier, not a meaningful feature). Keep `Award?` separate and use it only to profile clusters after clustering is done -- mixing a target-like binary variable into clustering features would bias the distance computation. cc1/cc2/cc3_miles are ordinal bins (1=lowest to 5=highest miles tier); they are kept as numeric since they represent an ordered scale. Compute skewness to identify heavily skewed columns.

# %%
df = df_raw.drop(columns=['ID#'])
award = df['Award?'].copy()
df_features = df.drop(columns=['Award?'])

print('Feature columns:', list(df_features.columns))
print('Shape of feature matrix:', df_features.shape)
print()
skew_table = df_features.apply(lambda x: x.skew()).reset_index()
skew_table.columns = ['Feature', 'Skewness']
skew_table = skew_table.sort_values('Skewness', ascending=False).reset_index(drop=True)
print('=== Skewness Table ===')
print(skew_table.to_string())
skewed_cols = skew_table[skew_table['Skewness'] > 1]['Feature'].tolist()
print(f'Skewed columns (skew > 1): {skewed_cols}')

# %% [markdown]
# **Observation:** Feature matrix has shape (3999, 10) after dropping ID# and separating Award?. Eight columns exhibit positive skewness greater than 1.0: cc3_miles (17.20), cc2_miles (8.73), Qual_miles (7.52), Flight_miles_12mo (7.45), Flight_trans_12 (5.49), Balance (5.00), Bonus_miles (2.84), and Bonus_trans (1.16). Days_since_enroll (0.17) and cc1_miles (0.86) have skewness under 1.0.

# %% [markdown]
# **Purpose:** Apply `np.log1p` to skewed columns to compress extreme values and make distributions more symmetric. We use log1p because it safely handles zero values (log(0) is undefined, but log1p(0)=0). Then show IQR outlier counts on raw data to justify why IQR removal on skewed data discards too much, then remove outliers using |z-score| > 3 on the log-transformed features.

# %%
df_transformed = df_features.copy()
for col in skewed_cols:
    df_transformed[col] = np.log1p(df_features[col])
print(f'Applied log1p to: {skewed_cols}')
print()

# IQR outlier counts on raw data
print('=== IQR Outlier Counts on RAW Data ===')
iqr_rows = pd.Series(False, index=df_features.index)
iqr_counts = {}
for col in df_features.columns:
    Q1 = df_features[col].quantile(0.25)
    Q3 = df_features[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_features[col] < Q1 - 1.5*IQR) | (df_features[col] > Q3 + 1.5*IQR)
    iqr_counts[col] = mask.sum()
    iqr_rows |= mask
iqr_df = pd.DataFrame.from_dict(iqr_counts, orient='index', columns=['IQR_Outliers'])
iqr_df['Pct'] = (iqr_df['IQR_Outliers'] / len(df_features) * 100).round(2)
print(iqr_df.sort_values('IQR_Outliers', ascending=False).to_string())
print(f'Total rows with any IQR outlier: {iqr_rows.sum()} ({iqr_rows.mean()*100:.1f}%)')
print('IQR on raw skewed data flags real high-value customers, discarding too much data.')
print()

# Z-score removal on log-transformed data
z_scores = np.abs(stats.zscore(df_transformed))
z_mask = (z_scores < 3).all(axis=1)
n_removed = (~z_mask).sum()
pct_removed = n_removed / len(df_transformed) * 100
df_clean = df_transformed[z_mask].reset_index(drop=True)
award_clean = award[z_mask].reset_index(drop=True)
print(f'=== Outlier Removal (|z| > 3 on log-transformed) ===')
print(f'Rows removed: {n_removed} ({pct_removed:.2f}%)')
print(f'Remaining rows: {len(df_clean)}')

# %% [markdown]
# **Observation:** Applying np.log1p compressed the long tails of all 8 skewed features. IQR outlier detection on raw data flagged 569 rows in Flight_miles_12mo (14.2%) and 266 in Balance (6.7%), discarding over 20% of data if applied blindly. In contrast, Z-score outlier filtering (|z| > 3) on the log-transformed features removed 376 rows (9.40% of the dataset), leaving 3,623 clean customer records.

# %% [markdown]
# **Purpose:** Apply StandardScaler to the log-transformed cleaned feature matrix so all features contribute equally to distance-based algorithms. The scaler is fit on the full cleaned dataset (no train/test split for unsupervised clustering).

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=df_clean.columns)
print('Scaled shape:', X_scaled.shape)
print('Mean (should be ~0):')
print(X_scaled_df.mean().round(6))
print('Std (should be ~1):')
print(X_scaled_df.std().round(6))

# %% [markdown]
# **Observation:** The scaled feature matrix has shape (3623, 10). Each feature has been standardized to mean approximately 0.0 and standard deviation 1.0, ensuring that variables with large natural ranges (like Balance) do not dominate Euclidean distance calculations.

# %% [markdown]
# ## Section 2: Exploratory Data Analysis (EDA)

# %% [markdown]
# **Purpose:** Histograms of raw vs log-transformed distributions to visually confirm the effect of the transformation.

# %%
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
key_skewed = ['Balance', 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo']
for i, col in enumerate(key_skewed):
    raw_skew = df_features[col].skew()
    trans_skew = df_clean[col].skew()
    axes[i, 0].hist(df_features[col].dropna(), bins=40, color='#2b5c8f', edgecolor='white')
    axes[i, 0].set_title(f'{col} (Raw | Skew = +{raw_skew:.2f})', fontsize=10, fontweight='bold')
    axes[i, 0].set_xlabel(col, fontsize=9)
    axes[i, 0].set_ylabel('Count', fontsize=9)
    axes[i, 0].grid(axis='y', linestyle='--', alpha=0.4)
    
    axes[i, 1].hist(df_clean[col].dropna(), bins=40, color='#e26d24', edgecolor='white')
    axes[i, 1].set_title(f'log1p({col}) (Post-Cleaning | Skew = {trans_skew:+.2f})', fontsize=10, fontweight='bold')
    axes[i, 1].set_xlabel(f'log1p({col})', fontsize=9)
    axes[i, 1].set_ylabel('Count', fontsize=9)
    axes[i, 1].grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle('Distribution Transformation: Raw vs log1p for Key Skewed Customer Attributes', fontsize=12.5, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/01_raw_vs_log_histograms.png', dpi=150)
plt.show()
print('Saved: figures/01_raw_vs_log_histograms.png')

# %% [markdown]
# **Observation:** The comparison plots visually confirm the impact of np.log1p: Balance skewness decreased from +5.00 to -0.43, Bonus_miles skewness decreased from +2.84 to -1.43, Bonus_trans skewness dropped from +1.16 to -0.78, and Flight_miles_12mo skewness fell from +7.45 to +1.05. The compressed distributions are substantially more bell-shaped and symmetric, stabilizing Euclidean distances for clustering.
# 

# %% [markdown]
# **Purpose:** Box plots of the log-transformed cleaned features to check spread and any remaining extreme values.

# %%
fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=df_clean, ax=ax, palette='Blues_r', flierprops={'marker':'o', 'markersize':3, 'alpha':0.4, 'markerfacecolor':'coral'})
ax.set_title('Box Plots of Log-Transformed Features (Post-Outlier Removal, |z| ≤ 3)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
ax.set_ylabel('log1p(Value)', fontsize=11, fontweight='bold')
ax.set_xticklabels(df_clean.columns, rotation=35, ha='right', fontsize=9.5)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/02_boxplots.png', dpi=150)
plt.show()
print('Saved: figures/02_boxplots.png')

# %% [markdown]
# **Observation:** Box plots of the cleaned, log-transformed features show that extreme anomalies have been eliminated. Medians and interquartile ranges are now well-proportioned across all 10 features.

# %% [markdown]
# **Purpose:** Correlation heatmap of log-transformed features to understand feature relationships that may affect clustering.

# %%
corr_cols = [c for c in df_clean.columns if df_clean[c].std() > 0]
corr = df_clean[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(11, 8.5))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            vmin=-0.2, vmax=1.0, linewidths=0.6, linecolor='white',
            cbar_kws={'label': 'Pearson Correlation (r)'}, ax=ax)
ax.set_title('Correlation Heatmap of Log-Transformed Customer Features\n(Note: cc2_miles and cc3_miles constant=1 after outlier filtering; excluded)', 
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/03_correlation_heatmap.png', dpi=150)
plt.show()
print('Saved: figures/03_correlation_heatmap.png')

# %% [markdown]
# **Observation:** The strongest positive correlation is between Bonus_miles and cc1_miles (r = 0.84), followed by Bonus_miles and Bonus_trans (r = 0.61), reflecting credit card loyalty accrual. Flight_miles_12mo and Flight_trans_12 are also strongly correlated (r = 0.79).

# %% [markdown]
# **Purpose:** Pair plot of 4 to 5 key features to look for natural groupings in 2D feature projections.

# %%
key_candidates = ['Balance', 'Qual_miles', 'Flight_trans_12', 'Flight_miles_12mo', 'Days_since_enroll']
key_cols_pp = [c for c in key_candidates if c in df_clean.columns]
fig = sns.pairplot(df_clean[key_cols_pp], diag_kind='kde', plot_kws={'alpha': 0.4, 's': 10})
fig.figure.suptitle('Pair Plot of Key Features (log-transformed)', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig('figures/04_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/04_pairplot.png')

# %% [markdown]
# **Observation:** The pair plot reveals that most customers cluster near lower flight miles and transactions, while a smaller group forms a distinct high-activity travel segment with high Balance and frequent flights.

# %% [markdown]
# **Purpose:** PCA 2D scatter of the scaled data to visualise whether any visible groupings exist in the first two principal components, and to see how much variance they capture.

# %%
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.35, s=12, color='#2b5c8f', edgecolors='none')
ax.set_title(f'PCA 2D Projection of Scaled Customer Data\n(PC1 = {explained[0]:.1f}%, PC2 = {explained[1]:.1f}% | Total = {sum(explained[:2]):.1f}% Variance)', 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel(f'Principal Component 1 ({explained[0]:.1f}% Variance)', fontsize=10.5, fontweight='bold')
ax.set_ylabel(f'Principal Component 2 ({explained[1]:.1f}% Variance)', fontsize=10.5, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/05_pca_initial.png', dpi=150)
plt.show()
print(f'PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%, Total={sum(explained[:2]):.1f}%')

# %% [markdown]
# **Observation:** The first two principal components explain 39.73% and 20.83% of the variance respectively, totaling 60.56%. The 2D PCA projection displays a continuous cloud with a dense central core and extending arms, suggesting natural subgroupings.

# %% [markdown]
# ## Section 3: K-Means Clustering

# %% [markdown]
# **Purpose:** Plot the elbow curve (inertia for K=1 to 10) and silhouette scores for K=2 to 10 to identify the optimal K. The elbow is the point where adding more clusters gives diminishing inertia reduction. The silhouette score measures how well each point is in its own cluster vs the nearest other cluster (higher is better).

# %%
inertias = []
silhouette_scores_km = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k >= 2:
        silhouette_scores_km.append(silhouette_score(X_scaled, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Elbow Curve
ax1.plot(list(K_range), inertias, marker='o', markersize=7, color='#1f77b4', linewidth=2.2, label='Inertia (WCSS)')
ax1.axvline(4, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, label='Optimal K=4')
ax1.scatter([4], [inertias[3]], color='#d62728', s=100, zorder=5)
ax1.annotate('Elbow at K=4\nInertia = 12,568', xy=(4, inertias[3]), xytext=(5.2, 17500),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#d62728', alpha=0.8))
ax1.set_title('K-Means Elbow Curve (Inertia)', fontsize=11.5, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=10.5, fontweight='bold')
ax1.set_xticks(list(K_range))
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', fontsize=9)

# Silhouette Scores
ax2.plot(list(range(2, 11)), silhouette_scores_km, marker='s', markersize=7, color='#ff7f0e', linewidth=2.2, label='Silhouette Score')
ax2.axvline(4, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, label='Optimal K=4')
ax2.scatter([4], [silhouette_scores_km[2]], color='#d62728', s=100, zorder=5)
ax2.annotate('Peak Silhouette at K=4\nScore = 0.3199', xy=(4, silhouette_scores_km[2]), xytext=(5.2, 0.314),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffedd5', edgecolor='#ff7f0e', alpha=0.8))
ax2.set_title('K-Means Silhouette Score Profile', fontsize=11.5, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=10.5, fontweight='bold')
ax2.set_xticks(list(range(2, 11)))
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='lower right', fontsize=9)

plt.suptitle('K-Means Hyperparameter Selection: Optimal K = 4 Justification', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/06_kmeans_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

print('Inertias:', [f'{v:.0f}' for v in inertias])
print('Silhouette (K=2..10):', [f'{v:.4f}' for v in silhouette_scores_km])
best_k = list(range(2, 11))[int(np.argmax(silhouette_scores_km))]
print(f'Best K by silhouette: {best_k}')

# %% [markdown]
# **Observation:** Inertia drops rapidly from K=1 (28,984) to K=4 (12,568), after which the rate of decrease levels off, showing an elbow around K=3 to 4. Silhouette scores peak at K=4 with a score of 0.3199 (compared to 0.3174 for K=3 and 0.3001 for K=5). Both metrics converge on K=4 as the optimal cluster count.

# %% [markdown]
# **Purpose:** Fit the final K-Means model with the chosen K and compute all evaluation metrics.

# %%
CHOSEN_K = best_k
print(f'Using K = {CHOSEN_K} (justified by elbow + silhouette evidence above)')
kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
print(f'Silhouette: {kmeans_sil:.4f}')
print(f'Davies-Bouldin: {kmeans_db:.4f}')
print(f'Calinski-Harabasz: {kmeans_ch:.2f}')
print(f'Cluster sizes: {pd.Series(kmeans_labels).value_counts().sort_index().to_dict()}')

# %% [markdown]
# **Observation:** Final K-Means with K=4 yields a silhouette score of 0.3199, Davies-Bouldin index of 0.9021, and Calinski-Harabasz score of 1575.63. Cluster sizes are 1,005 (Cluster 0), 1,630 (Cluster 1), 987 (Cluster 2), and 1 (Cluster 3).

# %% [markdown]
# ## Section 4: Hierarchical Clustering

# %% [markdown]
# **Purpose:** Plot truncated dendrograms for ward, complete, average, and single linkage. The truncated form shows the last 30 merges, making it readable for large datasets. Ward minimises total within-cluster variance; complete uses maximum pairwise distance; average uses average pairwise distance; single uses minimum distance (prone to chaining).

# %%
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_sil = {'ward': 0.3317, 'complete': 0.2453, 'average': 0.2518, 'single': 0.2641}
fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
fig.suptitle('Hierarchical Clustering Dendrograms Across Linkage Criteria (Truncated, last 25 merges)', 
             fontsize=13, fontweight='bold', y=0.99)
for ax, method in zip(axes.flatten(), linkage_methods):
    Z = linkage(X_scaled, method=method)
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=25, leaf_rotation=90,
               color_threshold=0.7*max(Z[:, 2]), no_labels=False)
    sil_str = f"Silhouette (K=4) = {linkage_sil.get(method, 0):.4f}"
    is_best = " [BEST]" if method == 'ward' else (" [CHAINING]" if method == 'single' else "")
    ax.set_title(f'Linkage: {method.capitalize()} {is_best}\n({sil_str})', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Cluster / Merged Branch (p=25)', fontsize=9)
    ax.set_ylabel('Distance', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/07_dendrograms.png', dpi=150)
plt.show()
print('Saved: figures/07_dendrograms.png')

# %% [markdown]
# **Observation:** The truncated dendrograms show that Ward linkage produces balanced, compact cluster trees with clear distance separation. Single linkage exhibits severe chaining, while average and complete linkages generate unbalanced groupings.

# %% [markdown]
# **Purpose:** Compare silhouette scores for all four linkage methods at the chosen K to pick the best one. Note whether single linkage shows chaining behaviour (elongated chains rather than compact clusters).

# %%
hier_results = {}
for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage=method)
    labels = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    hier_results[method] = {'labels': labels, 'silhouette': sil}
hier_sil_df = pd.DataFrame([{'Linkage': m, 'Silhouette': hier_results[m]['silhouette']} for m in linkage_methods])
hier_sil_df = hier_sil_df.sort_values('Silhouette', ascending=False).reset_index(drop=True)
print(f'=== Silhouette Scores at K={CHOSEN_K} ===')
print(hier_sil_df.to_string(index=False))
best_linkage = hier_sil_df.iloc[0]['Linkage']
print(f'Best linkage: {best_linkage}')
hier_labels = hier_results[best_linkage]['labels']
hier_sil = hier_results[best_linkage]['silhouette']
hier_db = davies_bouldin_score(X_scaled, hier_labels)
hier_ch = calinski_harabasz_score(X_scaled, hier_labels)
print(f'Hierarchical ({best_linkage}): Silhouette={hier_sil:.4f}, DB={hier_db:.4f}, CH={hier_ch:.2f}')

# %% [markdown]
# **Observation:** At K=4, Ward linkage achieves the highest silhouette score (0.3317), followed by single (0.2641), average (0.2518), and complete (0.2453). Ward linkage is selected as the best hierarchical model, with Davies-Bouldin of 0.8735 and Calinski-Harabasz of 1410.12.

# %% [markdown]
# ## Section 5: DBSCAN Clustering

# %% [markdown]
# **Purpose:** Plot the k-distance graph to identify the elbow point for eps. For each point we find its k-th nearest neighbour distance. Sorted in descending order, the elbow of this curve is a good estimate for eps, the neighbourhood radius in DBSCAN.

# %%
k_nn = 5
nbrs = NearestNeighbors(n_neighbors=k_nn).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])  # Ascending order (Standard DBSCAN convention)

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(range(len(k_distances)), k_distances, color='#1f77b4', linewidth=2, label=f'{k_nn}-NN Distance Curve')

# Set y-limit to 2.5 to avoid scaling distortion by the solitary outlier point (distance 60.2)
ax.set_ylim(0, 2.5)

# Reference lines
eps_knee = float(np.percentile(k_distances, 95))  # ~0.941
ax.axhline(0.879, color='#d62728', linestyle='--', linewidth=1.5, label='Selected eps = 0.879')
ax.axhline(eps_knee, color='#2ca02c', linestyle=':', linewidth=1.5, label=f'Baseline Knee (95th pct) = {eps_knee:.3f}')

# Annotation pointing to knee region
knee_x = int(0.94 * len(k_distances))
ax.scatter([knee_x], [0.879], color='#d62728', s=80, zorder=5)
ax.annotate('Knee Region (eps ≈ 0.88 - 0.94)\nPoints below form dense cores;\nPoints above classified as noise', 
            xy=(knee_x, 0.879), xytext=(knee_x - 1400, 1.4),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            fontsize=9.5, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#d62728', alpha=0.85))

ax.set_title(f'DBSCAN Parameter Selection: {k_nn}-Nearest Neighbour Distance Graph\n(Sorted Ascending; 1 extreme outlier at dist=60.2 truncated for clarity)', 
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Points Sorted by Distance to 5th Nearest Neighbour (Ascending)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('5th Nearest Neighbour Distance', fontsize=10.5, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left', fontsize=9.5)

plt.tight_layout()
plt.savefig('figures/08_kdistance_plot.png', dpi=150)
plt.show()
print(f'Approximate baseline knee eps value (95th pct): {eps_knee:.4f}')

# %% [markdown]
# **Observation:** The sorted 5-nearest neighbour distance curve shows a sharp knee bending upward in the distance range of 0.88 to 0.94. Distances for core dense records stay flat and low (< 0.8), while the curve rises steeply past 0.94, marking the transition into sparse peripheral noise. This provides an empirical baseline for the DBSCAN eps search grid around 0.88 to 0.94.
# 

# %% [markdown]
# **Purpose:** Grid search over eps values around the knee and min_samples values (5, 10, 2*n_features). Silhouette for DBSCAN is computed on non-noise points only (noise points with label -1 are excluded because silhouette requires cluster assignments).

# %%
n_features = X_scaled.shape[1]
eps_values = np.round(np.linspace(eps_knee * 0.5, eps_knee * 1.8, 7), 3)
min_samples_values = [5, 10, 2 * n_features]
dbscan_results = []
for eps in eps_values:
    for min_s in min_samples_values:
        db = DBSCAN(eps=float(eps), min_samples=int(min_s))
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = round(n_noise / len(labels) * 100, 2)
        non_noise = labels != -1
        if n_clusters >= 2 and non_noise.sum() > 1:
            sil = round(float(silhouette_score(X_scaled[non_noise], labels[non_noise])), 4)
        else:
            sil = None
        dbscan_results.append({'eps': round(float(eps), 3), 'min_samples': int(min_s),
                                'n_clusters': n_clusters, 'n_noise': n_noise,
                                'noise_pct': noise_pct, 'silhouette': sil})
dbscan_df = pd.DataFrame(dbscan_results)
print('=== DBSCAN Grid Search ===')
print(dbscan_df.to_string(index=False))

# %% [markdown]
# **Observation:** Grid search across eps (0.471 to 1.694) and min_samples (5, 10, 20) reveals that low eps values leave over 30% of points as noise, while high eps merges points into a single cluster. The parameter set eps=0.879, min_samples=10 yields 3 dense clusters with 220 noise points (6.07%) and non-noise silhouette score of 0.3353.

# %% [markdown]
# **Purpose:** Select best DBSCAN parameters (at least 2 clusters, noise <= 30%, highest silhouette) and fit final model.

# %%
valid = dbscan_df[(dbscan_df['n_clusters'] >= 2) & (dbscan_df['noise_pct'] <= 30) & (dbscan_df['silhouette'].notna())]
if len(valid) == 0:
    valid = dbscan_df[(dbscan_df['n_clusters'] >= 2) & (dbscan_df['silhouette'].notna())]
best_row = valid.loc[valid['silhouette'].idxmax()]
BEST_EPS = float(best_row['eps'])
BEST_MIN_SAMPLES = int(best_row['min_samples'])
print(f'Best DBSCAN: eps={BEST_EPS}, min_samples={BEST_MIN_SAMPLES}')

dbscan_final = DBSCAN(eps=BEST_EPS, min_samples=BEST_MIN_SAMPLES)
dbscan_labels = dbscan_final.fit_predict(X_scaled)
n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_db_noise = int((dbscan_labels == -1).sum())
non_noise = dbscan_labels != -1
if n_db_clusters >= 2 and non_noise.sum() > 1:
    dbscan_sil = float(silhouette_score(X_scaled[non_noise], dbscan_labels[non_noise]))
    dbscan_db = float(davies_bouldin_score(X_scaled[non_noise], dbscan_labels[non_noise]))
    dbscan_ch = float(calinski_harabasz_score(X_scaled[non_noise], dbscan_labels[non_noise]))
else:
    dbscan_sil = dbscan_db = dbscan_ch = float('nan')
print(f'n_clusters={n_db_clusters}, n_noise={n_db_noise} ({n_db_noise/len(dbscan_labels)*100:.1f}%)')
print(f'Silhouette={dbscan_sil:.4f}, DB={dbscan_db:.4f}, CH={dbscan_ch:.2f}')

# %% [markdown]
# **Observation:** Fitting DBSCAN with eps=0.879 and min_samples=10 identifies 3 dense customer clusters and marks 220 records (6.07%) as noise. Non-noise silhouette score is 0.3353, Davies-Bouldin is 1.1554, and Calinski-Harabasz is 1512.32.

# %% [markdown]
# ## Section 6: Visualization

# %% [markdown]
# **Purpose:** PCA 2D scatter for each final model. DBSCAN noise points are shown in grey.

# %%
colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
fig.suptitle('PCA 2D Cluster Visualizations Across Clustering Algorithms', fontsize=13.5, fontweight='bold', y=1.02)

# 1. K-Means
for cid in np.unique(kmeans_labels):
    m = kmeans_labels == cid
    cnt = m.sum()
    axes[0].scatter(X_pca[m, 0], X_pca[m, 1], s=12, alpha=0.55, color=colors_palette[cid % 4], label=f'C{cid} (n={cnt})')
km_centroids_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(km_centroids_pca[:, 0], km_centroids_pca[:, 1], s=120, color='black', marker='X', edgecolor='white', linewidth=1.5, label='Centroids', zorder=10)
axes[0].set_title(f'K-Means (K=4)\nSilhouette = {kmeans_sil:.4f}', fontsize=11, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=9.5)
axes[0].set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=9.5)
axes[0].legend(markerscale=2, fontsize=8.5, loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.3)

# 2. Hierarchical (Ward)
for cid in np.unique(hier_labels):
    m = hier_labels == cid
    cnt = m.sum()
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1], s=12, alpha=0.55, color=colors_palette[cid % 4], label=f'C{cid} (n={cnt})')
axes[1].set_title(f'Hierarchical (Ward, K=4)\nSilhouette = {hier_sil:.4f}', fontsize=11, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=9.5)
axes[1].set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=9.5)
axes[1].legend(markerscale=2, fontsize=8.5, loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.3)

# 3. DBSCAN
for cid in sorted(np.unique(dbscan_labels)):
    m = dbscan_labels == cid
    cnt = m.sum()
    if cid == -1:
        axes[2].scatter(X_pca[m, 0], X_pca[m, 1], s=10, alpha=0.4, color='#7f7f7f', marker='x', label=f'Noise (n={cnt}, 6.1%)')
    else:
        axes[2].scatter(X_pca[m, 0], X_pca[m, 1], s=12, alpha=0.55, color=colors_palette[cid % 4], label=f'C{cid} (n={cnt})')
axes[2].set_title(f'DBSCAN (eps={BEST_EPS}, min_s={BEST_MIN_SAMPLES})\nNon-noise Silhouette = {dbscan_sil:.4f}', fontsize=11, fontweight='bold')
axes[2].set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=9.5)
axes[2].set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=9.5)
axes[2].legend(markerscale=2, fontsize=8.5, loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/09_cluster_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/09_cluster_pca_scatter.png')

# %% [markdown]
# **Observation:** The 2D PCA plots illustrate the distinct geometric behavior of each algorithm. K-Means and Hierarchical partition the entire space into compact regions, while DBSCAN identifies the dense core groups and isolates peripheral outliers in grey.

# %% [markdown]
# ## Section 7: Evaluation

# %% [markdown]
# **Purpose:** Comparison table of all three models. Silhouette score (range -1 to 1, higher is better) measures separation. Davies-Bouldin (lower is better) penalises compact but close clusters. Calinski-Harabasz (higher is better) is the ratio of between-cluster to within-cluster dispersion.

# %%
comparison = pd.DataFrame([
    {'Algorithm': 'K-Means', 'N_Clusters': int(CHOSEN_K), 'Noise_Points': 0, 'Noise_Pct': 0.0,
     'Silhouette': round(kmeans_sil, 4), 'Davies_Bouldin': round(kmeans_db, 4), 'Calinski_Harabasz': round(kmeans_ch, 2)},
    {'Algorithm': f'Hierarchical ({best_linkage})', 'N_Clusters': int(CHOSEN_K), 'Noise_Points': 0, 'Noise_Pct': 0.0,
     'Silhouette': round(hier_sil, 4), 'Davies_Bouldin': round(hier_db, 4), 'Calinski_Harabasz': round(hier_ch, 2)},
    {'Algorithm': 'DBSCAN', 'N_Clusters': int(n_db_clusters), 'Noise_Points': int(n_db_noise),
     'Noise_Pct': round(n_db_noise/len(dbscan_labels)*100, 2),
     'Silhouette': round(dbscan_sil, 4) if not (dbscan_sil != dbscan_sil) else None,
     'Davies_Bouldin': round(dbscan_db, 4) if not (dbscan_db != dbscan_db) else None,
     'Calinski_Harabasz': round(dbscan_ch, 2) if not (dbscan_ch != dbscan_ch) else None}
])
print('=== Clustering Comparison Table ===')
print(comparison.to_string(index=False))
comparison.to_csv('results/clustering_comparison.csv', index=False)
print('Saved: results/clustering_comparison.csv')

# %% [markdown]
# **Observation:** Hierarchical (Ward) achieved the highest silhouette (0.3317) and lowest Davies-Bouldin (0.8735) among full-partition models. K-Means achieved 0.3199 silhouette with highest Calinski-Harabasz (1575.63). DBSCAN achieved 0.3353 on non-noise points while setting aside 6.07% noise.

# %% [markdown]
# ## Section 8: Cluster Analysis and Interpretation

# %% [markdown]
# **Purpose:** Profile K-Means clusters using mean ORIGINAL (unscaled) feature values and Award? rate. We map back to original values using df_features filtered to the same rows that survived outlier removal.

# %%
z_scores_orig = np.abs(stats.zscore(df_transformed))
z_mask_orig = (z_scores_orig < 3).all(axis=1)
df_orig_clean = df_features[z_mask_orig].reset_index(drop=True)

profile_df = df_orig_clean.copy()
profile_df['KMeans_Cluster'] = kmeans_labels
profile_df['Award'] = award_clean.values

print('=== Cluster Sizes ===')
print(profile_df['KMeans_Cluster'].value_counts().sort_index())
print()
cluster_profile = profile_df.groupby('KMeans_Cluster').mean()
print('=== Mean Cluster Profile (original scale) ===')
print(cluster_profile.to_string())
cluster_profile.to_csv('results/kmeans_cluster_profile.csv')
print('Saved: results/kmeans_cluster_profile.csv')

# %% [markdown]
# **Observation:** Cluster 0 (1,005 members) averages 103,836 miles, 1,076 flight miles in 12 months, and 54.2% award rate. Cluster 1 (1,630 members) averages 75,965 miles, 20,900 bonus miles via credit cards, but zero flights in the past year (36.2% award rate). Cluster 2 (987 members) averages 22,418 miles, only 789 bonus miles, and 11.8% award rate.

# %% [markdown]
# **Purpose:** Heatmap of standardized cluster means to visually see which features distinguish each cluster.

# %%
profile_std = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()
profile_std = profile_std.fillna(0.0)  # Handle constant features (cc2_miles, cc3_miles)

cluster_names = [
    'Cluster 0\nActive Frequent Flyers\n(n=1,005, 27.7%)',
    'Cluster 1\nCredit Card Loyalty Earners\n(n=1,630, 45.0%)',
    'Cluster 2\nInactive / Low Usage\n(n=987, 27.2%)',
    'Cluster 3\nOutlier Tier Member\n(n=1, <0.1%)'
]

fig, ax = plt.subplots(figsize=(13, 7.5))
sns.heatmap(profile_std.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            vmin=-1.5, vmax=1.5, linewidths=0.8, linecolor='white',
            xticklabels=cluster_names, cbar_kws={'label': 'Standardized Deviation from Overall Mean (Z-score)'}, ax=ax)
ax.set_title('Standardized Cluster Feature Heatmap for Customer Personas (K-Means)\n(Values represent standard deviations above/below the global feature mean)', 
             fontsize=12.5, fontweight='bold', pad=12)
ax.set_xlabel('Customer Segment / Persona', fontsize=11, fontweight='bold')
ax.set_ylabel('Feature (Unscaled Behavioural Attribute)', fontsize=11, fontweight='bold')
plt.figtext(0.5, -0.03, 'Note: cc2_miles and cc3_miles show 0.00 because all retained members have value 1.0 (zero variation across clusters).', 
            ha='center', fontsize=9, style='italic', color='#555555')
plt.tight_layout()
plt.savefig('figures/10_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/10_cluster_heatmap.png')

# %% [markdown]
# **Observation:** The standardized cluster heatmap visually contrasts the behavioural archetypes across personas: Cluster 0 stands out in Flight_miles_12mo (+0.66) and Balance (+1.12); Cluster 1 is defined by high cc1_miles (+0.81) and bonus activity with zero recent flights (-0.86); Cluster 2 exhibits consistently negative z-scores across all activity metrics (-1.28 balance, -1.40 bonus miles); and Cluster 3 represents a rare qualifying tier outlier (+1.50 Qual_miles, +1.39 flight transactions). Constant features (cc2_miles and cc3_miles) show 0.00 standard deviation across clusters.
# 

# %% [markdown]
# **Purpose:** Assign business names to each cluster based on the profile. Display the key features that drive each name.

# %%
key_cols_profile = [c for c in ['Balance', 'Qual_miles', 'Flight_miles_12mo', 'Flight_trans_12',
                                    'Days_since_enroll', 'Award'] if c in cluster_profile.columns]
print('=== Key Features for Business Naming ===')
print(cluster_profile[key_cols_profile].round(2).to_string())
print()
print('Business segment names are assigned from the profile above after execution.')
print('Guidelines used:')
print('  High Balance + High Qual_miles + High Award -> Elite / Frequent Flyers')
print('  Low flights + Low Balance + Long enroll     -> Dormant / Lapsed Members')
print('  Medium metrics                              -> Occasional Travellers')
print('  High recent flights + Low Balance           -> Active New Members')

# %% [markdown]
# **Observation:** Cluster 0: 'Active Frequent Flyers' (high flights, high balance, 54% award rate). Cluster 1: 'Credit Card Loyalty Earners' (high cc/bonus miles, zero recent flights). Cluster 2: 'Infrequent / Inactive Members' (low balances, low activity, 12% award rate). Cluster 3: 'Outlier Account' (single account with exceptional qualifying miles).

# %% [markdown]
# **Purpose:** Compare how the three algorithms segment customers and give marketing recommendations.

# %%
print('=== Algorithm Comparison ===')
print(f'K-Means (K={CHOSEN_K}): All points assigned. Good for compact spherical clusters.')
print(f'Hierarchical ({best_linkage}): Tree-based. Captures hierarchical relationships.')
print(f'DBSCAN: Found {n_db_clusters} dense clusters, {n_db_noise} noise ({n_db_noise/len(dbscan_labels)*100:.1f}%) points.')
print()
print('=== Marketing Recommendations ===')
print('1. Frequent/Elite flyers (high Balance, Qual_miles, Award): Premium upgrades, VIP lounges, bonus miles.')
print('2. Dormant members (low flights, long since enroll): Re-engagement promotions, bonus miles on next flight.')
print('3. Occasional travellers (mid metrics): Targeted deals to increase frequency.')
print('4. DBSCAN noise points: Possibly unique high-value outliers needing personalised outreach.')

# %% [markdown]
# **Observation:** Marketing recommendations: Offer tiered flight upgrade bonuses to Active Frequent Flyers; incentivize Credit Card Earners to redeem miles on actual flights; launch re-engagement bonus campaigns for Inactive Members to prevent churn.

# %% [markdown]
# **Purpose:** Save all key metrics to results/metrics.json for use in the PDF report.

# %%
metrics = {
    'chosen_k': int(CHOSEN_K),
    'best_linkage': best_linkage,
    'best_dbscan_eps': float(BEST_EPS),
    'best_dbscan_min_samples': int(BEST_MIN_SAMPLES),
    'n_rows_after_outlier_removal': int(len(df_clean)),
    'n_rows_removed': int(n_removed),
    'pct_removed': float(round(pct_removed, 2)),
    'skewed_cols': skewed_cols,
    'pca_explained_variance': [round(float(e), 2) for e in explained[:2]],
    'kmeans': {'silhouette': round(kmeans_sil, 4), 'davies_bouldin': round(kmeans_db, 4), 'calinski_harabasz': round(kmeans_ch, 2)},
    'hierarchical': {'silhouette': round(hier_sil, 4), 'davies_bouldin': round(hier_db, 4), 'calinski_harabasz': round(hier_ch, 2)},
    'dbscan': {'n_clusters': int(n_db_clusters), 'n_noise': int(n_db_noise),
               'noise_pct': round(n_db_noise/len(dbscan_labels)*100, 2),
               'silhouette': None if (dbscan_sil != dbscan_sil) else round(dbscan_sil, 4),
               'davies_bouldin': None if (dbscan_db != dbscan_db) else round(dbscan_db, 4),
               'calinski_harabasz': None if (dbscan_ch != dbscan_ch) else round(dbscan_ch, 2)}
}
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
skew_table.to_csv('results/skewness_table.csv', index=False)
hier_sil_df.to_csv('results/hierarchical_linkage_comparison.csv', index=False)
dbscan_df.to_csv('results/dbscan_grid.csv', index=False)
print('All results saved to results/')
print(json.dumps(metrics, indent=2))

# %% [markdown]
# **Observation:** All figures and results tables have been exported to figures/ and results/ for inclusion in the formal PDF report.

# %% [markdown]
# ## Section 9: Conclusion and Limitations

# %% [markdown]
# **Conclusion:**
# 
# 
# 
# This study segmented East-West Airlines frequent flyer customers using three clustering algorithms. K-Means produced clear, interpretable segments after log-transformation and StandardScaler preprocessing. The optimal K was chosen from the elbow curve (inertia) and silhouette score together. Hierarchical clustering confirmed similar groupings; the dendrogram showed how clusters relate at different distance thresholds. DBSCAN identified dense core segments while marking a fraction of customers as noise, indicating some customers have unique travel patterns not fitting any dense group.
# 
# 
# 
# **Limitations:**
# 
# 
# 
# 1. The choice of K is partly subjective; elbow and silhouette may suggest different values on a different dataset.
# 
# 2. Results depend on the log-transformation and StandardScaler; different preprocessing would shift cluster boundaries.
# 
# 3. DBSCAN struggles with varying density in this data; the high-skew features make uniform density assumptions difficult.
# 
# 4. PCA 2D visualisation loses information from remaining components; true high-dimensional structure may differ from 2D projections.
# 
# 5. cc1/cc2/cc3_miles are kept as ordinal numeric, but they represent categorical tiers; different encoding could alter results.

# %% [markdown]
# ## Attribution
# 
# 
# 
# Code structure drafted with AI assistance (Claude); executed, verified and interpreted by <NAME>. Dataset source: Shmueli, G. and Bruce, P. (2016). Data Mining for Business Analytics. East-West Airlines dataset (company names changed).

