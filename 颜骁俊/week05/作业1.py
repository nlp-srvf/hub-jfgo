import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


def calculate_intra_cluster_distances(X, labels, metric='average'):
    """Calculate intra-cluster distance for each cluster"""
    unique_labels = np.unique(labels)
    cluster_distances = {}

    for label in unique_labels:
        cluster_samples = X[labels == label]
        n_samples = len(cluster_samples)

        if n_samples <= 1:
            cluster_distances[label] = 0.0
            continue

        distances = euclidean_distances(cluster_samples)

        if metric == 'average':
            total_distance = np.sum(distances) / 2  # Avoid double counting
            avg_distance = total_distance / (n_samples * (n_samples - 1) / 2)
            cluster_distances[label] = avg_distance
        elif metric == 'max':
            max_distance = np.max(distances)
            cluster_distances[label] = max_distance
        else:
            raise ValueError("metric must be 'average' or 'max'")

    return cluster_distances


def sort_clusters_by_intra_distance(cluster_distances, ascending=True):
    """Sort clusters by intra-cluster distance"""
    sorted_items = sorted(cluster_distances.items(),
                          key=lambda x: x[1],
                          reverse=not ascending)

    sorted_clusters = [item[0] for item in sorted_items]
    sorted_distances = [item[1] for item in sorted_items]

    return sorted_clusters, sorted_distances


def visualize_cluster_distances(sorted_clusters, sorted_distances):
    """Visualize sorted intra-cluster distances"""
    plt.figure(figsize=(10, 6))
    plt.bar([f'Cluster {c}' for c in sorted_clusters], sorted_distances, color='skyblue', edgecolor='black')
    plt.xlabel('Clusters')
    plt.ylabel('Intra-cluster Distance')
    plt.title('Clusters Sorted by Intra-cluster Distance')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    n_features = 2

    # Create 3 clusters with different compactness
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(100, n_features))
    cluster2 = np.random.normal(loc=[5, 5], scale=1.0, size=(100, n_features))
    cluster3 = np.random.normal(loc=[10, 0], scale=1.5, size=(100, n_features))

    X = np.vstack([cluster1, cluster2, cluster3])

    # Perform KMeans clustering
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # Calculate intra-cluster average distance
    intra_distances_avg = calculate_intra_cluster_distances(X, labels, metric='average')
    print("Intra-cluster Average Distance:")
    for cluster, distance in intra_distances_avg.items():
        print(f"Cluster {cluster}: {distance:.4f}")

    # Sort clusters by average distance
    sorted_clusters_avg, sorted_distances_avg = sort_clusters_by_intra_distance(intra_distances_avg)
    print("\nClusters sorted by average intra-distance:", sorted_clusters_avg)

    # Calculate intra-cluster max distance
    intra_distances_max = calculate_intra_cluster_distances(X, labels, metric='max')
    print("\nIntra-cluster Maximum Distance:")
    for cluster, distance in intra_distances_max.items():
        print(f"Cluster {cluster}: {distance:.4f}")

    # Sort clusters by max distance
    sorted_clusters_max, sorted_distances_max = sort_clusters_by_intra_distance(intra_distances_max)
    print("\nClusters sorted by maximum intra-distance:", sorted_clusters_max)

    # Visualize results
    visualize_cluster_distances(sorted_clusters_avg, sorted_distances_avg)


if __name__ == "__main__":
    main()
