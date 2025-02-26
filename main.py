from pathlib import Path
import matplotlib.pyplot as plt
from cluster_it import kmeans,sum_of_squared_errors, random_index, load_data, plot_clusters

def main():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    
    plt.figure(figsize=(8, 6))
    plt.title("Original Spiral")
    
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], color="brg"[true_labels[i] % 3], edgecolor='k')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(plots_dir / "original_dataset.png")
    plt.show()
    
    options = ["euclidean", "cosine", "l3"]
    
    for option in options:
        best_sum_error = float("inf") 
        best_random_index= 0
        best_run = None
        results = []
        runs = 10
        
        for run in range(runs):
            print(f"Metric: {option} - Run {run+1}/{runs}")

            predicted, centroids = kmeans(data, k=3, metric=option)
            sum_sqd_error = sum_of_squared_errors(data, predicted, centroids, metric=option)

            random_idx = random_index(true_labels, predicted)
            results.append((run, sum_sqd_error, random_idx, predicted, centroids))
            print(f"sum of squared errors: {sum_sqd_error:.4f} with random index: {random_idx:.4f}")
            
            if sum_sqd_error < best_sum_error and random_idx > best_random_index:
                best_sum_error = sum_sqd_error
                best_random_index = random_idx
                best_run = run

        print(f"\nbest run: {best_run+1}")
        print(f"best sum_sqd_error: {best_sum_error:.4f}, best random index: {best_random_index:.4f}")
        
        for run, sum_sqd_error, random_idx, predicted, centroids in results:
            plt.figure(figsize=(8, 6))
            plt.title(f"kmeans {run+1} - metric: {option} {sum_sqd_error:.1f} - random index: {random_idx:.1f}")
            plot_clusters(data, predicted, centroids)
            plt.savefig(plots_dir / f"{option}_{run+1}.png")
            plt.show()

if __name__ == "__main__":
    main()
