# Module 5: Unsupervised Learning

## Introduction

Today marks a significant shift in how we think about machine learning.

In Modules 2 through 4, we always had a target variable—sales, churn, fraud. We had labels, and we trained models to predict those labels.

**Now we throw that away. No labels. No target variable.**

Unsupervised learning is about discovering structure in data when you don't know what you're looking for. You're exploring, not predicting.

This might sound less useful, but unsupervised learning solves critical business problems: customer segmentation, anomaly detection, data visualization, feature extraction. These are problems where labels don't exist or are too expensive to obtain.

**Validating unsupervised learning**: "Right" is about usefulness, not correctness. Use internal metrics (silhouette, inertia), check stability across runs, and—most importantly—validate with domain experts. Do clusters suggest actionable strategies? A "statistically optimal" 7-cluster solution that marketing can't operationalize is less useful than a 3-cluster solution they can act on.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the difference between supervised and unsupervised learning
2. **Apply** K-means and DBSCAN clustering algorithms and interpret results
3. **Determine** optimal number of clusters using elbow method and silhouette scores
4. **Apply** PCA for dimensionality reduction and interpret principal components
5. **Use** manifold learning techniques (t-SNE, UMAP) for visualization
6. **Identify** business applications for clustering and dimensionality reduction

---

## 5.1 Clustering

### Supervised vs Unsupervised

| Supervised | Unsupervised |
|------------|--------------|
| Have labels | No labels |
| Learn to predict | Discover structure |
| Regression, Classification | Clustering, Dim. Reduction |

**Supervised**: "Here are the right answers; learn to predict them."

**Unsupervised**: "Here's the data; find interesting patterns."

### Three Components: K-Means

Even unsupervised algorithms fit our three-component framework:

| Component | K-Means |
|-----------|---------|
| **Decision Model** | Cluster assignments — each point belongs to nearest centroid |
| **Quality Measure** | Within-cluster sum of squares (inertia) |
| **Update Method** | Iterative assignment-update — alternate between assigning and moving centroids |

**Key difference from supervised learning**: Without labels, we define "quality" differently. Instead of prediction error, we measure how compact and well-separated clusters are.

**Distinguishing real structure from noise**: Clustering algorithms will always find clusters—even in random data. Use the gap statistic (compares quality to random data), stability analysis (cluster on subsets—real structure is stable), and multiple algorithms (if K-means, DBSCAN, and hierarchical all find similar groups, structure is more credible). Always verify clusters predict something meaningful.

### Clustering Applications

- **Customer segmentation** — Group by purchasing behavior, target marketing per segment
- **Document grouping** — Organize by topic without predefined categories
- **Anomaly detection** — Find observations that don't fit any group
- **Image compression** — Reduce color palettes by clustering similar colors
- **Gene expression** — Group genes with similar activation patterns

### K-Means Algorithm

**The algorithm:**
1. **Choose K** (number of clusters)
2. **Randomly initialize** K centroids
3. **Assign**: Each point to nearest centroid
4. **Update**: Move centroids to mean of assigned points
5. **Repeat** until centroids stop moving

**The objective:**

$$\text{minimize } \sum_{i=1}^{K}\sum_{x \in C_i} ||x - \mu_i||^2$$

Where $\mu_i$ is the centroid of cluster $C_i$. Minimize total distance from points to their centroids.

**Strengths:**
- Fast and scalable—works on millions of points
- Easy to implement and interpret
- Works well with spherical clusters

**Weaknesses:**
- Must specify K in advance
- Sensitive to initialization
- Assumes spherical, similar-sized clusters

**"Spherical" clusters**: K-means assigns points to the nearest centroid using Euclidean distance, implicitly assuming clusters are ball-shaped with equal spread in all directions. K-means essentially draws Voronoi cells (straight-line boundaries)—any cluster that can't fit in a convex cell will be problematic. For non-spherical shapes, use DBSCAN (any shape), GMMs (elliptical), or spectral clustering (complex manifolds).

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times, keep best
    random_state=42
)

labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
print(f"Inertia: {kmeans.inertia_}")
```

### Choosing K: Elbow Method

**Process:**
1. Run K-means for K = 1, 2, 3, ..., n
2. Plot inertia vs K
3. Look for the "elbow" where adding clusters gives diminishing returns

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

### Choosing K: Silhouette Score

For each point, measure how similar it is to its own cluster vs. other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

**Interpretation:**
- s = 1: Well-clustered (far from other clusters)
- s = 0: On boundary between clusters
- s = -1: Probably in wrong cluster

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall score
score = silhouette_score(X_scaled, labels)

# Per-sample (for diagnostics)
sample_scores = silhouette_samples(X_scaled, labels)
```

**Individual scores are useful too:**
- Find misclassified points (negative scores)
- Identify boundary cases (scores near 0)
- Detect outliers (very low scores)

**Business consideration**: Sometimes the "right" K comes from domain knowledge, not just metrics!

**When elbow and silhouette disagree**: Elbow (inertia) measures compactness; silhouette measures both compactness AND separation. Adding clusters always reduces inertia but may not improve silhouette if new clusters aren't well-separated. If elbow says 5 and silhouette says 3, clusters 4-5 might be subdividing natural groups. Look at both metrics, examine cluster profiles, consider business constraints, and check stability. There's rarely a single "correct" K.

### DBSCAN: Density-Based Clustering

K-means assumes spherical clusters. DBSCAN handles:
- Irregular shapes
- Different densities
- Noise/outliers

**Core concepts:**
- **Core point**: Has at least `min_samples` points within `eps` distance
- **Border point**: Within `eps` of a core point, but not core itself
- **Noise point**: Neither (labeled -1)

**Algorithm:**
1. Find all core points
2. Connect core points within `eps` of each other (transitively)
3. Assign border points to nearest core point's cluster
4. Everything else is noise

**Connecting core points (step 2) in detail:**

Two core points belong to the same cluster if they're "density-reachable":
- Direct: Within `eps` of each other
- Transitive: A connects to B, B connects to C → A and C same cluster

This is graph traversal where core points are nodes and edges exist between points within `eps`. Each connected component becomes a cluster.

**Choosing eps and min_samples**: For eps, use the k-distance plot—compute each point's distance to its k-th nearest neighbor, sort and plot, look for the elbow. For min_samples, start with dimensions + 1 or 2×dimensions. Larger min_samples = more conservative. If DBSCAN parameter tuning is frustrating, try HDBSCAN—it removes the eps parameter entirely.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise: {n_noise}")
```

### K-Means vs DBSCAN

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| # Clusters | Must specify | Auto-detected |
| Shapes | Spherical | Arbitrary |
| Handles noise | No | Yes (labels -1) |
| Speed | Very fast | Slower |

### HDBSCAN: Hierarchical DBSCAN

HDBSCAN addresses DBSCAN's sensitivity to the `eps` parameter:

| Aspect | DBSCAN | HDBSCAN |
|--------|--------|---------|
| Parameters | `eps` and `min_samples` | Just `min_samples` |
| Cluster densities | Assumes uniform | Handles varying |

**How it works:**
1. Build a hierarchy considering all possible `eps` values
2. Find stable clusters that persist across `eps` range
3. Extract flat clustering from the hierarchy

```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(X)
```

### Hierarchical Clustering

Build a tree of nested clusters.

**Agglomerative (bottom-up):**
1. Start: Each point is its own cluster
2. Find two closest clusters
3. Merge them
4. Repeat until one cluster remains

**Linkage methods** (distance between clusters):
- **Single**: Minimum distance between any points
- **Complete**: Maximum distance between any points
- **Average**: Average distance between all pairs
- **Ward**: Minimize variance increase when merging

**Dendrogram**: Tree showing merge history
- Y-axis = distance at which clusters merged
- Cut at any height to get that many clusters

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Cut to get 3 clusters
labels = fcluster(Z, t=3, criterion='maxclust')
```

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "There's one correct number of clusters" | Clustering is exploratory. Multiple valid solutions exist. Business context matters. |
| "K-means always works" | Fails on complex shapes, varying densities, outliers. |
| "Silhouette = 0.9 means perfect clustering" | Silhouette measures separation, not business meaning. |
| "More clusters is always better" | Reduces variance but may not be useful. Aim for interpretable segments. |

---

## 5.2 Dimensionality Reduction

### Why Reduce Dimensions?

**The curse of dimensionality:**
- High-dimensional spaces are sparse
- Distance metrics become less meaningful
- Models overfit more easily

**Benefits:**
- **Visualization**: Can't plot 50 dimensions. Can plot 2.
- **Noise reduction**: Remove uninformative dimensions
- **Faster training**: Fewer features
- **Feature extraction**: Create meaningful composites

**Are we losing important information?** You ARE losing information—the question is signal vs. noise. If 10 components capture 95% of variance, the last 40 combined contribute 5% (mostly noise). Verify by comparing model performance with/without reduction. Caveats: rare but important patterns may have low variance; PCA doesn't know your target, so captured variance might not be predictive.

### Principal Component Analysis (PCA)

Find new axes (principal components) that:
1. Are linear combinations of original features
2. Capture maximum variance
3. Are orthogonal (uncorrelated)

**Algorithm:**
1. Center the data (subtract mean)
2. Find direction of maximum variance → PC1
3. Find direction of max remaining variance, perpendicular to PC1 → PC2
4. Continue...

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
```

### Choosing Number of Components

- **Scree plot**: Variance explained vs component number
- **Cumulative variance**: Keep enough for 80-95%
- **Kaiser criterion**: Keep components with eigenvalue > 1

```python
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative variance plot
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
```

### Interpreting PCA Loadings

**Loadings** show how original features contribute to each component:

| Feature  | PC1 | PC2 |
|----------|-----|-----|
| Income   | 0.8 | 0.1 |
| Age      | 0.7 | -0.2 |
| Spending | 0.6 | 0.8 |

**Interpretation:**
- PC1 loads on Income, Age, Spending → "Overall affluence"
- PC2 loads mainly on Spending → "Spending tendency"

**Naming is subjective**: Component naming is interpretation, not discovery. Two analysts might name the same loadings differently ("Wealth" vs "Financial Stability" vs "Affluence Score"). Report actual loadings alongside interpretation, acknowledge subjectivity, and validate with domain experts. If you can't tell a coherent story, the component may not be meaningfully interpretable.

```python
# Loadings are in components_ (rows = components, cols = features)
loadings = pca.components_

import polars as pl
loadings_df = pl.DataFrame(
    loadings,
    schema=feature_names
).with_row_index("PC")
```

### t-SNE for Visualization

PCA assumes linear relationships. t-SNE handles non-linear manifolds.

**Goal**: Preserve local neighborhoods in 2D
- Points close in high-D stay close
- Points far apart can move freely

**Key parameter**: `perplexity` (~5-50)
- Roughly expected number of neighbors
- Try multiple values

**Critical caveats:**
- Stochastic—different runs give different results
- **Cluster sizes are meaningless** (distances distorted)
- Can create false patterns in random data
- Slow for large datasets
- **ONLY for visualization, NOT preprocessing!**

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

**Never use t-SNE coordinates as features for a classifier.** Distances are distorted. Use PCA for preprocessing.

**Trusting t-SNE clusters**: t-SNE preserves local neighborhoods but distorts global distances, cluster sizes, and densities. To avoid being fooled: run multiple times with different seeds/perplexity, validate with clustering on the original high-D data (if K-means finds no structure there, t-SNE may be misleading), and check perplexity sensitivity. t-SNE is for visualization and hypothesis generation—always verify clusters with methods on the original data.

### UMAP

UMAP (Uniform Manifold Approximation and Projection) is often better than t-SNE:

- **Faster**, especially for large data
- **Preserves global structure** better
- **Can be used for preprocessing** (not just visualization)
- **More reproducible**

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)
```

### Method Comparison

| Method | Speed | Global Structure | Use For |
|--------|-------|------------------|---------|
| PCA | Fast | Preserved | Preprocessing, visualization |
| t-SNE | Slow | Lost | Visualization only |
| UMAP | Medium | Partially preserved | Both |

**Rule of thumb:**
- PCA for preprocessing and quick visualization
- t-SNE or UMAP for beautiful visualizations
- UMAP if you want the best of both worlds

### MNIST Example

- Original: 784 dimensions (28×28 pixels)
- PCA to 2D: Blurry separation
- t-SNE/UMAP to 2D: Clear digit clusters!

**Why?** The digit manifold is non-linear. PCA's linear assumption can't capture it. t-SNE and UMAP can.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "PCA finds the most important features" | PCA finds linear combinations. Components may not correspond to individual features. |
| "t-SNE cluster sizes are meaningful" | t-SNE distorts distances. A big cluster in t-SNE might be same size as small one in reality. |
| "More components = better" | More preserves more info but may include noise. Choose based on task. |
| "Dimensionality reduction always helps ML models" | Sometimes original features are better. Compare performance. |

---

## Reflection Questions

1. You're segmenting customers for marketing. K-means suggests 5 clusters, but your team can only create 3 campaigns. What do you do?

2. Your clustering puts 95% of data in one cluster and creates 4 tiny ones. Is this a problem? What might cause this?

3. When would you choose DBSCAN over K-means? Give a business example.

4. A colleague says they found "the optimal number of clusters." Why should you be skeptical?

5. PCA on customer data shows PC1 explains 80% of variance. Should you only use PC1?

6. You run t-SNE twice and get different-looking plots. Is one wrong?

7. A colleague says "UMAP proves our data has 5 clusters." What's wrong with this statement?

---

## Practice Problems

1. Given cluster assignments, calculate silhouette score by hand for a small example

2. Interpret PCA loadings for a business dataset (name the components)

3. Choose between K-means and DBSCAN for different data scenarios

4. Explain why t-SNE shouldn't be used for preprocessing

5. For customer data with features {Income, Age, Transactions, Days_Since_Purchase}, describe what PC1 and PC2 might represent

---

## Chapter Summary

**Six key takeaways from Module 5:**

1. **Unsupervised learning** discovers structure without labels

2. **K-means** is fast but needs spherical clusters and specified K

3. **DBSCAN** handles arbitrary shapes and identifies outliers

4. **Silhouette scores** measure cluster quality (but not business meaning)

5. **PCA** finds linear combinations that maximize variance

6. **t-SNE/UMAP** reveal non-linear structure—use t-SNE for visualization only!

---

## What's Next

In Module 6, we tackle **Neural Networks Fundamentals**:
- Perceptrons and multi-layer networks
- Activation functions
- Backpropagation
- Deep learning basics

Here's an interesting connection: dimensionality reduction is related to neural network feature learning. Neural networks automatically learn compressed representations of inputs—that's partly why deep learning works so well.
