from pathlib import Path
import matplotlib.pyplot as plt
from cluster_it import cut_merge,cophenetic_coef, silhouette_score,plot_dendrogram,compute_centroids,hierarchical_clustering, kmeans,sum_of_squared_errors, random_index, load_data, plot_clusters

def test_kmeans(data, true_labels):
    plt.figure(figsize=(8, 6))
    plt.title("Original Spiral")
    
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], color="brg"[true_labels[i] % 3], edgecolor='k')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(Path("plots") / "original_dataset.png")
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
            plt.savefig(Path("plots/part1") / f"{option}_{run+1}.png")
            plt.show()

# hierarchical teting function.
def test_hierarchical(data, true_labels):
    metrics = ["euclidean", "cosine", "l3"]
    linkage_methods = ["single", "complete", "average", "centroid"]
    
    best_sse = float("inf")
    best_ccc = -float("inf")
    best_sil = -float("inf")
    best_rii= 0
    best_run = None
    runs = 3

    for run in range(runs):
        for metric in metrics:
            for linkage in linkage_methods:
                print(f"Hierarchical: Metric: {metric} | Linkage: {linkage}  | Run {run+1}/{runs}")
                
                merge_history = hierarchical_clustering(data, metric=metric, linkage_method=linkage)
                predicted = cut_merge(len(data), merge_history, k=3)

                centroids = compute_centroids(data, predicted)
                sse = sum_of_squared_errors(data, predicted, centroids, metric=metric)
                ri = random_index(true_labels, predicted)
                ccc = cophenetic_coef(data, merge_history, metric=metric)
                silhouette = silhouette_score(data, predicted, metric=metric)

                # to save the absolute best run I set up this hirarchical... since we doing hierarchical clustering
                # cophenetic correlation coefficient is the most important metric to consider,
                # this is followed by silhouette score, then random index and finally sse.
                if ccc > best_ccc:
                    best_ccc = ccc
                    best_sil = silhouette
                    best_rii = ri
                    best_sse = sse
                    best_run = (run, metric, linkage, predicted)

                elif ccc == best_ccc:
                    if silhouette > best_sil:
                        best_sil = silhouette
                        best_rii = ri
                        best_sse = sse
                        best_run = (run, metric, linkage, predicted)
                        
                    elif silhouette == best_sil:
                        if ri > best_rii:
                            best_rii = ri
                            best_sse = sse
                            best_run = (run, metric, linkage, predicted)
                        
                        elif ri == best_rii:
                            if sse < best_sse:
                                best_sse = sse
                                best_run = (run, metric, linkage, predicted)

                print(f"Run {run+1} | Metric: {metric}\nLinkage: {linkage}")
                print(f"SSE: {sse:.4f} | Rand Index: {ri:.4f}")
                print(f"Cophenetic Corr Coef: {ccc:.4f} | Silhouette: {silhouette:.4f}")
                print(f"Merges: {len(merge_history)}")
                
                plt.figure(figsize=(8, 6))
                plt.title(f"Hierarchical Clusters (k=3) | Run: {run+1} | {metric}, {linkage}")
                plot_clusters(data, predicted, centroids)
                save_path = Path("plots/part2") / f"hierarchical_{run+1}_{metric}_{linkage}.png"
                plt.savefig(save_path)
                plt.close()

                plot_dendrogram(merge_history, samples=len(data), path=Path("plots/part2") / f"dendrogram_{run+1}_{metric}_{linkage}.png")

    if best_run is not None:
        run, metric, linkage, best_labels = best_run
        print(f"\nBest Hierarchical Run Details:\n")
        print(f"Run: {run+1} | Metric: {metric} | Linkage: {linkage}")
        print(f"Best SSE: {best_sse:.4f} | Best Rand Index: {best_rii:.4f}")
        print(f"Best Cophenetic Corr Coef: {best_ccc:.4f} | Best Silhouette Score: {best_sil:.4f}")
        
        plt.figure(figsize=(8, 6))
        plt.title(f"Best Overall Run: {run+1}, Metric: {metric}, Linkage: {linkage}")
        best_centroids = compute_centroids(data, best_labels)
        plot_clusters(data, best_labels, best_centroids)
        plt.savefig(Path("plots/part2") / f"best_hierarchical_{run+1}_{metric}_{linkage}.png")
        plt.close()

def main():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    data_file = Path(__file__).parent / "spiral-dataset.csv"
    data, true_labels = load_data(data_file)
    
    # Run k-means test.
    # test_kmeans(data, true_labels)
    # Run hierarchical clustering test.
    test_hierarchical(data, true_labels)
    
if __name__ == "__main__":
    main()
