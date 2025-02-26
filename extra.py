import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from cluster_it import plot_clusters, random_index


def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    return X, true_labels

def run_sklearn_dbscan(eps=1.5, min_samples=5, metric="euclidean", scale_data=False):
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    data = np.array(data)

    if scale_data:
        data = StandardScaler().fit_transform(data)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    pred_labels = db.fit_predict(data)
    unique_labels = set(pred_labels)

    if -1 in unique_labels:
        unique_labels.remove(-1)

    centroids = []
    for label in unique_labels:
        cluster_points = data[np.where(pred_labels == label)]
        centroids.append(cluster_points.mean(axis=0))

    centroids = np.array(centroids) if centroids else np.empty((0, data.shape[1]))
    
    plt.figure(figsize=(8, 6))
    scale_str = "(scaled)" if scale_data else ""
    plt.title(f"Sklearn DBSCAN {scale_str} (eps={eps}, min_samples={min_samples})")
    plot_clusters(data, pred_labels, centroids)
    plt.show()

    ri = random_index(true_labels, pred_labels)
    print(f"Sklearn DBSCAN Rand Index: {ri:.4f}")

def dbscan_parameter_search(metric="euclidean", scale_data=False):
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    data = np.array(data)

    if scale_data:
        data = StandardScaler().fit_transform(data)

    eps_values = [0.5, 1.0, 1.5, 2.0]
    min_samples_values = [3, 5, 8, 10]
    best_ri = 0
    best_params = (None, None)

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms, metric=metric)
            pred_labels = db.fit_predict(data)
            ri = random_index(true_labels, pred_labels)

            print(f"eps={eps}, min_samples={ms} => RI={ri:.4f}")
            if ri > best_ri:
                best_ri = ri
                best_params = (eps, ms)

    print("\nbest DBSCAN parms:")
    print(f"  eps={best_params[0]}, min_samples={best_params[1]}, Rand Index={best_ri:.4f}")

def run_sklearn_spectral(n_clusters=3, affinity='rbf', gamma=None, n_neighbors=None, scale_data=False):
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    data = np.array(data)

    if scale_data:
        data = StandardScaler().fit_transform(data)

    sc_kwargs = {"n_clusters": n_clusters, "affinity": affinity, "random_state": 127}

    if affinity == "rbf" and gamma is not None:
        sc_kwargs["gamma"] = gamma

    if affinity == "nearest_neighbors" and n_neighbors is not None:
        sc_kwargs["n_neighbors"] = n_neighbors

    sc = SpectralClustering(**sc_kwargs)
    pred_labels = sc.fit_predict(data)
    centroids = []

    for label in range(n_clusters):
        cluster_points = data[np.where(pred_labels == label)]

        if len(cluster_points) > 0:
            centroids.append(cluster_points.mean(axis=0))
        else:
            centroids.append(np.zeros(data.shape[1]))

    centroids = np.array(centroids)

    scale_str = "(scaled)" if scale_data else ""
    title_str = f"Sklearn Spectral Clustering {scale_str} (affinity={affinity}"
    
    if gamma is not None and affinity == "rbf":
        title_str += f", gamma={gamma}"

    if n_neighbors is not None and affinity == "nearest_neighbors":
        title_str += f", n_neighbors={n_neighbors}"

    title_str += ")"

    plt.figure(figsize=(8, 6))
    plt.title(title_str)
    plot_clusters(data, pred_labels, centroids)
    plt.show()

    ri = random_index(true_labels, pred_labels)

    print(f"Sklearn Spectral Clustering Rand Index: {ri:.4f}")

def run_sklearn_hierarchical(n_clusters=3, linkage="ward", metric="euclidean", scale_data=False):
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    data = np.array(data)

    if scale_data:
        data = StandardScaler().fit_transform(data)
    if linkage == "ward":
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
    
    pred_labels = hc.fit_predict(data)
    centroids = []

    for label in range(n_clusters):
        cluster_points = data[np.where(pred_labels == label)]
        if len(cluster_points) > 0:
            centroids.append(cluster_points.mean(axis=0))
        else:
            centroids.append(np.zeros(data.shape[1]))

    centroids = np.array(centroids)

    plt.figure(figsize=(8, 6))
    scale_str = "(scaled)" if scale_data else ""
    plt.title(f"Sklearn Hierarchical Clustering {scale_str} (linkage={linkage})")
    plot_clusters(data, pred_labels, centroids)
    plt.show()
    ri = random_index(true_labels, pred_labels)

    print(f"Sklearn Hierarchical Clustering Rand Index: {ri:.4f}")


if __name__ == "__main__":
    print("Using dbscan with ecluidian distance:\n")
    run_sklearn_dbscan(eps=1.5, min_samples=5, metric="euclidean", scale_data=False)
    # param search
    dbscan_parameter_search(metric="euclidean", scale_data=False)

    print("Using rbf as the affinity:\n")
    run_sklearn_spectral(n_clusters=3, affinity='rbf', gamma=0.1)

    print("Using nearest_neighbors as the affinity:\n")
    run_sklearn_spectral(n_clusters=3, affinity='nearest_neighbors', n_neighbors=10)

    print("Using hierarchical clustering with ward linkage:\n")
    run_sklearn_hierarchical(n_clusters=3, linkage="ward", metric="euclidean", scale_data=False)
