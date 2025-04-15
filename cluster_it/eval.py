import math

# Took reference from the sckit learn base.py where they have s score function which 
# calculates sum of squared errors
# https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/base.py#L619


# left here from part 1 and for reference
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

def compute_centroids(data, labels):

    # find unique clusters using set
    unique_labels = list(set(labels))
    centroids = []

    for ul in unique_labels:
        # get teh points in this cluster
        points = []
        for i in range(len(data)):
            if labels[i] == ul:
                points.append(data[i])

        # compute mean/centroid 
        dim = len(data[0])
        centroid = [0.0] * dim

        for point in points:
            for d in range(dim):
                centroid[d] += point[d]

        for d in range(dim):
            centroid[d] /= len(points)

        centroids.append(centroid)

    return centroids

def compute_distance(p1, p2, metric="euclidean"):

    if metric == "euclidean":
        euc_sum = 0

        for a, b in zip(p1, p2):
            euc_sum += (a - b) ** 2

        euc_sqrt = math.sqrt(euc_sum)

        return euc_sqrt
    
    elif metric == "cosine":
        dot = 0
        sum1 = 0
        sum2 = 0

        for a, b in zip(p1, p2):
            dot += a * b

        for a in p1:
            sum1 += a ** 2
        
        for b in p2:
            sum2 += b ** 2

        norm1 = math.sqrt(sum1)
        norm2 = math.sqrt(sum2)

        # add small number to avoid division by zero
        cosine_sim = dot / (norm1 * norm2 + 1e-9)
        cosine_simmilarity = 1 - cosine_sim
        
        return cosine_simmilarity
    
    elif metric == "l3":
        # L3 norm
        # sum of absolute differences cubed
        sum_cubed = 0
        for a, b in zip(p1, p2):
            sum_cubed += abs(a - b) ** 3
        
        l3 = sum_cubed ** (1.0 / 3.0)

        return l3
    
    else:
        raise ValueError("Unsupported metric: %s" % metric)

def silhouette_score(data, labels, metric="euclidean"):
    # a and b are avg dist to nearby clusters
    n = len(data)
    clusters = {}

    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    
    silhouettes = []
    
    for i in range(n):
        label = labels[i]
        
        same_cluster = clusters[label]
        if len(same_cluster) > 1:
            sum_distance = 0
            for j in same_cluster:
                distance = compute_distance(data[i], data[j], metric)
                sum_distance += distance   
            a = sum_distance / len(same_cluster) - 1
            
        else:
            a = 0
        
        # b is the lowest avg
        b = float("inf")
        for other, indices in clusters.items():
            if other == label:
                #self
                continue
            sum_distance = 0
            for j in indices:
                distance = compute_distance(data[i], data[j], metric)
                sum_distance += distance
            avg_distance = sum_distance / len(indices)
            b = min(b, avg_distance)
        
        if max(a, b) > 0:
            s = (b - a) / max(a, b)
        else:
            s = 0

        silhouettes.append(s)
    
    silhouettes_sum = sum(silhouettes)/n
    return silhouettes_sum

def cophenetic_coef(data, merge_history, metric="euclidean"):
    n = len(data)
    
    original_distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            original_distances[(i, j)] = compute_distance(data[i], data[j], metric)
    
    # cluster members with their indices as tuples
    cluster_members = {}
    for i in range(n):
        cluster_members[(i,)] = [i]

    cophenetic_distances = {}
    
    for merge in merge_history:
        # print(type(merge))
        # print(len(merge))
        cluster1, cluster2, merge_distance, new_cluster = merge
       
        members1 = cluster_members[cluster1]
        members2 = cluster_members[cluster2]
        new_members = members1 + members2
        
        for i in members1:
            for j in members2:
                if i < j:
                    key = (i, j)
                else:
                    key = (j, i)
                
                cophenetic_distances[key] = merge_distance
        
        cluster_members[new_cluster] = new_members
        
    originals = []
    cophenetic = []
    for key, dist in original_distances.items():
        originals.append(dist)
        cophenetic.append(cophenetic_distances.get(key, 0))
    
    mean_orig = sum(originals) / len(originals)
    mean_coph = sum(cophenetic) / len(cophenetic)
    
    num = 0
    for x, y in zip(originals, cophenetic):
        num += (x - mean_orig) * (y - mean_coph)
    
    denominator1 = 0
    denominator2 = 0
    for x in originals:
        denominator1 += (x - mean_orig) ** 2

    for y in cophenetic:
        denominator2 += (y - mean_coph) ** 2
    
    denominator1_sqrt = math.sqrt(denominator1)
    denominator2_sqrt = math.sqrt(denominator2)

    
    if denominator1_sqrt * denominator2_sqrt == 0:
        return 0
    
    final = num / (denominator1_sqrt * denominator2_sqrt)

    return final
