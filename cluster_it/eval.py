# Took reference from the sckit learn base.py where they have s score function which 
# calculates sum of squared errors
# https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/base.py#L619

def sum_of_squared_errors(data, labels, centroids, metric="euclidean"):
    total_sum = 0.0

    for i in range(len(centroids)):
        clusters = []

        for j in range(len(labels)):
            if labels[j] == i:
                clusters.append(data[j]) 

        #  the distance between each point and the centroid
        dists = []
        if metric == "euclidean":
            for point in clusters:
                diff = []
                for d in range(len(point)):
                    diff.append(point[d] - centroids[i][d])

                sum_squared = 0
                for d in diff:
                    sum_squared += d ** 2

                dists.append(sum_squared ** 0.5)

        elif metric == "cosine":
            # same as in the kmeans.py file
            for point in clusters:
                x_norm = 0
                for d in range(len(point)):
                    x_norm += point[d] ** 2
                x_norm = x_norm ** 0.5

                c_norm = 0
                for d in range(len(centroids[i])):
                    c_norm += centroids[i][d] ** 2
                c_norm = c_norm ** 0.5

                dot_product = 0
                for d in range(len(point)):
                    dot_product += point[d] * centroids[i][d]

                # division by zero is avvoided by adding a tiny number
                normalized_product = (x_norm * c_norm) + 1e-12
                cos_similarity = dot_product / normalized_product
                dists.append(1 - cos_similarity)

        else:
            # the only other we support is the L3 norm
            for point in clusters:
                diff = []
                
                for d in range(len(point)):
                    diff.append(abs(point[d] - centroids[i][d]))
                sum_cubed = 0
                for d in diff:
                    sum_cubed += d ** 3

                dists.append(sum_cubed ** (1.0 / 3.0))

        for d in dists:
            total_sum += d ** 2 

    return total_sum


def random_index(true_labels, predicted):
    count = 0
    total_pairs = len(true_labels) * (len(true_labels) - 1) / 2

    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):

            if (true_labels[i] == true_labels[j]) and (predicted[i] == predicted[j]):
                count += 1

            elif (true_labels[i] != true_labels[j]) and (predicted[i] != predicted[j]):
                count += 1

    # the rand index is the pairs that are either in the same cluster or not
    random_index = count / total_pairs
    return random_index
