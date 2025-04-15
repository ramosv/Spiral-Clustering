import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def load_data(file_path):
    data = []
    true_labels = []

    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 3:
                data.append([float(values[0]), float(values[1])])
                true_labels.append(int(values[2]))

    return data, true_labels

def plot_clusters(data, labels, centroids):
    colors = ["red", "green", "blue"]
    
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], color=colors[labels[i] % len(colors)], edgecolor='k', alpha=0.6)
    
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color="black", marker="X", s=200)
    
    plt.xlabel("X")
    plt.ylabel("Y")

def plot_dendrogram(merge_history, samples, path=None):
    linkage_matrix = convert_history(merge_history, samples)
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)

    plt.xlabel("Data Points in Clusters")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    # try to save it or just show it if no path is available

    plt.savefig(path) if path else plt.show()
    plt.close()
    #plt.show()

# merge histry is a list of tuples!
def convert_history(merge_history, samples):
    mapping = {}
    # a mapping of tuples, where each tuple is a cluster
    # and teh value is the cluster label
    for i in range(samples):
        mapping[(i,)] = i

    next_label = samples
    linkage = []
    
    for cluster1, cluster2, merge_dist, new_cluster in merge_history:
        cluster1 = tuple(sorted(cluster1))
        cluster2 = tuple(sorted(cluster2))
        new_cluster = tuple(sorted(new_cluster))

        label1 = mapping[cluster1]
        label2 = mapping[cluster2]

        size = len(new_cluster)
        linkage.append([label1, label2, merge_dist, size])
        mapping[new_cluster] = next_label

        next_label += 1

    matrix = np.array(linkage)

    return matrix
