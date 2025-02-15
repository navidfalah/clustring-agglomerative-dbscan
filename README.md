# 🧩 Clustering Techniques

Welcome to the **Clustering Techniques** repository! This project explores various clustering algorithms, including **K-Means**, **Agglomerative Clustering**, and **DBSCAN**. It also covers evaluation metrics and visualizations for clustering tasks.

---

## 📂 **Project Overview**

This repository demonstrates how to apply and evaluate clustering algorithms using **Scikit-learn**, **mglearn**, and **Matplotlib**. It includes:

- **K-Means Clustering**: Partitioning data into `k` clusters.
- **Agglomerative Clustering**: Hierarchical clustering using a bottom-up approach.
- **DBSCAN**: Density-based clustering to identify clusters of varying shapes.
- **Evaluation Metrics**: Adjusted Rand Index (ARI) and Silhouette Score.
- **Visualizations**: Cluster assignments, dendrograms, and face dataset clustering.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **mglearn**
- **NumPy**
- **Matplotlib**
- **SciPy**

---

## 📊 **Datasets**

The project uses the following datasets:
- **Synthetic Blobs**: For basic clustering examples.
- **Two Moons**: For non-linear clustering.
- **Labeled Faces in the Wild (LFW)**: For face clustering and visualization.

---

## 🧠 **Key Concepts**

### 1. **K-Means Clustering**
- Partitions data into `k` clusters by minimizing the within-cluster variance.
- Sensitive to initialization and the number of clusters (`k`).

### 2. **Agglomerative Clustering**
- Builds a hierarchy of clusters using a bottom-up approach.
- Visualized using dendrograms.

### 3. **DBSCAN**
- Identifies clusters based on density.
- Handles noise and clusters of varying shapes.

### 4. **Evaluation Metrics**
- **Adjusted Rand Index (ARI)**: Measures similarity between true and predicted clusters.
- **Silhouette Score**: Evaluates the quality of clustering.

### 5. **Face Clustering**
- Applies clustering to the LFW dataset for face grouping.

---

## 🚀 **Code Highlights**

### K-Means Clustering
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
```

### Agglomerative Clustering
```python
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
```

### DBSCAN
```python
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2)
```

### Dendrogram Visualization
```python
linkage_array = ward(X)
dendrogram(linkage_array)
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
```

### Face Clustering
```python
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels) + 1):
    mask = labels == cluster
    fig, axes = plt.subplots(1, np.sum(mask), figsize=(15, 4))
    for image, ax in zip(X_people[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/clustering-techniques.git
   cd clustering-techniques
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook clustering.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
