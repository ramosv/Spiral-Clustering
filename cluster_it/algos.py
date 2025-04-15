import random

#part 2, implement hierarchical clustering.
#refernece: https://scikit-learn.org/stable/modules/clustering.html#id11

"""
A key difference in hierarchical clustering is that rather than computing the distance between a k number of centroids,
we use all the points in the dataset to compute the distance between each point and the other points.
We have a few options for the distance metric, but the ones required in this assignment are:
- single: minimum distance between points in two clusters
- complete: maximum distance between points in two clusters
- average: average distance between points in two clusters
- centroid: distance between the centroids of two clusters
"""
def hierarchical_clustering(data, metric="euclidean", linkage_method="single"):

    clusters = []
    for i in range(len(data)):
        clusters.append([i])

    merge_history = []

    while len(clusters) > 1:
        min_dist = float("inf")
        to_merge = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = compute_distances( data, [clusters[i], clusters[j]], metric, linkage_method,"hierarchical")
                if d < min_dist:
                    min_dist = d
                    to_merge = (i, j)

        i, j = to_merge
        new_cluster = clusters[i] + clusters[j]

        
        cluster1 = tuple((clusters[i])) 
        cluster2 = tuple((clusters[j]))
        new = tuple(new_cluster)
        #we use this later for plotting the dendrogram
        merge_history.append((cluster1, cluster2, min_dist, new))

        # remove the two clusters and add the new one
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)

    return merge_history

def cut_merge(n_samples, merge_history, k):

    clusters = []
    
    for i in range(n_samples):
       clusters.append([i])
    
    merge_index = 0

    # iterate until we have k clusters
    while len(clusters) > k and merge_index < len(merge_history):
        # the tuples
        cluster1, cluster2, _, new_cluster = merge_history[merge_index]

        # instanciate the
        idx1, idx2 = None, None
        for idx, c in enumerate(clusters):
            if set(c) == set(cluster1):
                idx1 = idx
            elif set(c) == set(cluster2):
                idx2 = idx

        #  if both found merge
        if idx1 is not None and idx2 is not None:
            merged = list(new_cluster)
            # this is because we are removing the clusters from the list
            for remove in sorted([idx1, idx2], reverse=True):
                clusters.pop(remove)

            clusters.append(merged)

        merge_index += 1

    # empty clusters
    labels = [None] * n_samples

    #populate the labels
    for id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = id

    return labels

#Part1
# I took reference from the scikit learn KMeans implementation
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# It is a simplified implementtion with half of the parameters of the original implementation
# shource code https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/cluster/_kmeans.py#L1186

def kmeans(data, k=3, max_iter=100, tolerance=1e-4, metric="euclidean"):
    indices = random.sample(range(len(data)), k)
    centroids = []

    for i in indices:
        centroids.append(data[i])

    for iter in range(max_iter):
        distances = compute_distances(data, centroids, metric)

        # index of the minimum distance for each point
        pred_labels = []
        for dist in distances:
            min_index = 0
            min_value = dist[0]

            for i in range(1, len(dist)):
                if dist[i] < min_value:
                    min_value = dist[i]
                    min_index = i

            pred_labels.append(min_index)

        # labels to a list of clusters
        clusters = {}
        for i in range(k):
            clusters[i] = []
        for idx in range(len(pred_labels)):
            label = pred_labels[idx]
            clusters[label].append(data[idx])

        # getting the new centroids
        new_centroids = []
        for i in range(k):
            if len(clusters[i]) > 0:
                mean_centroid = []

                for dim in range(len(clusters[i][0])):
                    sum_dim = 0

                    for point in clusters[i]:
                        sum_dim += point[dim]

                    mean_dim = sum_dim / len(clusters[i])
                    mean_centroid.append(mean_dim)
            else:
                mean_centroid = data[random.randint(0, len(data) - 1)]
            new_centroids.append(mean_centroid)

        # calculate the movement of centroids
        step = 0
        for new, old in zip(new_centroids, centroids):
            diff_sum = 0
            for i in range(len(new)):
                diff_sum += (new[i] - old[i]) ** 2
            step += diff_sum ** 0.5 

        centroids = new_centroids

        if step < tolerance:
            break

    # now the final labels
    labels = []
    for point in data:
        dists = []
        
        for centroid in centroids:
            diff_sum = 0

            for i in range(len(point)):
                diff_sum += (point[i] - centroid[i]) ** 2
                
            dists.append(diff_sum ** 0.5)

        min_index = 0
        min_value = dists[0]

        for i in range(1, len(dists)):
            if dists[i] < min_value:
                min_value = dists[i]
                min_index = i

        labels.append(min_index)

    return labels, centroids

"""
Repurposed function from last assignment to this: so that it support both kmeans and hierarchical clustering
"""
def compute_distances(data, centroids, distance_metric=None, linkage=None, clustering="kmeans"):

    if clustering == "kmeans" and distance_metric is not None and linkage is None:
        samples = len(data)
        k = len(centroids)
        dists = []  
        
        for i in range(samples):
            row = []
            for j in range(k):
                row.append(0)
            dists.append(row)
        
        for i in range(k):
            if distance_metric == "euclidean":
                for j in range(samples):
                    sum_squared = 0
                    for d in range(len(data[j])):
                        diff = data[j][d] - centroids[i][d]
                        sum_squared = sum_squared + diff * diff
                    dists[j][i] = sum_squared ** 0.5

            elif distance_metric == "cosine":
                for j in range(samples):
                    x_norm = 0
                    for d in range(len(data[j])):
                        x_norm = x_norm + data[j][d] * data[j][d]
                    x_norm = x_norm ** 0.5

                    c_norm = 0
                    for d in range(len(centroids[i])):
                        c_norm = c_norm + centroids[i][d] * centroids[i][d]
                    c_norm = c_norm ** 0.5

                    dot_product = 0
                    for d in range(len(data[j])):
                        dot_product = dot_product + data[j][d] * centroids[i][d]

                    normalized_prod = (x_norm * c_norm) + 1e-12  # to avoid division by zero
                    cosine_sim = dot_product / normalized_prod
                    dists[j][i] = 1 - cosine_sim

            elif distance_metric == "l3":
                for j in range(samples):
                    sum_cubed = 0
                    for d in range(len(data[j])):
                        diff = abs(data[j][d] - centroids[i][d])
                        sum_cubed = sum_cubed + diff ** 3
                    dists[j][i] = sum_cubed ** (1.0 / 3.0)

        return dists

    elif clustering == "hierarchical":
        # In hierarchical mode, we assume that 'centroids' is really a list of clusters,
        # each represented as a list of indices into the original data.
        clusters = centroids  # Rename for clarity.
        if len(clusters) < 2:
            raise ValueError("Need at least two clusters for hierarchical distance computation.")
        
        # For this example, compute the distance only between the first two clusters:
        cluster1 = clusters[0]
        cluster2 = clusters[1]

        if linkage == "single":
            # Single linkage: minimum distance between any two points (one from each cluster).
            min_dist = float("inf")
            for i in range(len(cluster1)):
                index1 = cluster1[i]
                for j in range(len(cluster2)):
                    index2 = cluster2[j]
                    sum_squared = 0
                    for d in range(len(data[index1])):
                        diff = data[index1][d] - data[index2][d]
                        sum_squared = sum_squared + diff * diff
                    dist = sum_squared ** 0.5
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        elif linkage == "complete":
            max_dist = 0
            for i in range(len(cluster1)):
                index1 = cluster1[i]
                for j in range(len(cluster2)):
                    index2 = cluster2[j]
                    sum_squared = 0
                    for d in range(len(data[index1])):
                        diff = data[index1][d] - data[index2][d]
                        sum_squared = sum_squared + diff * diff
                    dist = sum_squared ** 0.5
                    if dist > max_dist:
                        max_dist = dist
            return max_dist

        elif linkage == "average":
            total_dist = 0
            count = 0
            for i in range(len(cluster1)):
                index1 = cluster1[i]
                for j in range(len(cluster2)):
                    index2 = cluster2[j]
                    sum_squared = 0
                    for d in range(len(data[index1])):
                        diff = data[index1][d] - data[index2][d]
                        sum_squared = sum_squared + diff * diff
                    dist = sum_squared ** 0.5
                    total_dist = total_dist + dist
                    count = count + 1
            if count == 0:
                return 0
            else:
                return total_dist / count

        elif linkage == "centroid":
            dim = len(data[0])
            centroid1 = [0.0] * dim
            centroid2 = [0.0] * dim

            for i in range(len(cluster1)):
                index1 = cluster1[i]
                for d in range(dim):
                    centroid1[d] = centroid1[d] + data[index1][d]
            for d in range(dim):
                centroid1[d] = centroid1[d] / len(cluster1)

            for j in range(len(cluster2)):
                index2 = cluster2[j]
                for d in range(dim):
                    centroid2[d] = centroid2[d] + data[index2][d]
            for d in range(dim):
                centroid2[d] = centroid2[d] / len(cluster2)

            sum_squared = 0
            for d in range(dim):
                diff = centroid1[d] - centroid2[d]
                sum_squared = sum_squared + diff * diff
            return sum_squared ** 0.5

        else:
            raise ValueError("Invalid linkage method")

    else:    
        raise ValueError("Unknown clustering type. Use 'kmeans' or 'hierarchical'.")
