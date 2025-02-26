import random
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

# Again I took reference from the scikit learn KMeans implementation
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# They have 3 functions that do something similar to this one
# fit_predict, fit and fit_transform
# This is a modification of fit_tranform while also taking the distance metric as a

def compute_distances(data, centroids, distance_metric):
    # both data and centroids are numpy arrays so we take the first dimension

    samples = len(data)
    k = len(centroids)
    dists = []

    for _ in range(samples):
        dists.append([0] * k)

    for i in range(k):
        # looking at sckit learn source code: https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/cluster/_kmeans.py#L1100
        # I found that they eucledian distance is their default distance metric

        if distance_metric == "euclidean":
            for j in range(samples):
                difference = []
                for d in range(len(data[j])):
                    difference.append(data[j][d] - centroids[i][d])

                sum_squared = 0
                for d in difference:
                    sum_squared += d ** 2

                dists[j][i] = sum_squared ** 0.5

        # for the cosine similarity I took reference from sckit learn again
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

        elif distance_metric == "cosine":
            # cosine similarity
            # normalizing the data and centroids
            # we normalize because we looking for the angle between the two vectors
            
            # axis 1 for the columns
            for j in range(samples):
                x_norm = 0
                for d in range(len(data[j])):
                    x_norm += data[j][d] ** 2

                x_norm = x_norm ** 0.5

                c_norm = 0
                for d in range(len(centroids[i])):
                    c_norm += centroids[i][d] ** 2

                c_norm = c_norm ** 0.5
                dot_product = 0

                for d in range(len(data[j])):
                    dot_product += data[j][d] * centroids[i][d]

                # sckit learn source code: https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/metrics/pairwise.py#L1686
                # we do 1e-12 to avoid division by zero, this is just a tiny number to avoid that
                # I was choked to see that they support over 22 distance metrics
                
                normalized_prod = (x_norm * c_norm) + 1e-12
                cosine_sim = dot_product / normalized_prod
                
                # cosine similarity is 1 - cosine similarity
                dists[j][i] = 1 - cosine_sim

        else:
            # l3 distance
            # anything other than euclidean and cosine will trigger this.
            for j in range(samples):
                diff = []
                for d in range(len(data[j])):
                    diff.append(abs(data[j][d] - centroids[i][d]))

                sum_cubed = 0
                for d in diff:
                    sum_cubed += d ** 3

                dists[j][i] = sum_cubed ** (1 / 3)

    # return the distances
    return dists
