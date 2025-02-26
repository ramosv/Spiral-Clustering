import matplotlib.pyplot as plt

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
