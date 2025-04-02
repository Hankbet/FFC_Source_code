import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA


class GFC:
    def __init__(self, theta=None, k_gene_force=100, k_assign=2, radio=75):
        self.theta = theta  
        self.k_gene_force = k_gene_force
        self.k_assign = k_assign
        self.radio = radio
        self.labels_ = None
    def fit(self, X):
        n_samples, n_features = X.shape
     
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.k_gene_force + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        gene_forces = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            neighbors = indices[i, 1:self.k_gene_force + 1]
            dists = distances[i, 1:self.k_gene_force + 1]

            
            diff = X[neighbors] - X[i]
            norms = np.linalg.norm(diff, axis=1)
            norms[norms == 0] = 1e-8  
            directions = diff / norms[:, np.newaxis]
            gene_forces[i] = np.sum(dists[:, np.newaxis] * directions, axis=0)
        gene_force_magnitudes = np.linalg.norm(gene_forces, axis=1)
        if self.theta is None:
            self.theta = np.percentile(gene_force_magnitudes, self.radio)
        edge_mask = gene_force_magnitudes > self.theta
        edge_indices = np.where(edge_mask)[0]
        edge_objects = X[edge_indices]
        if len(edge_indices) == 0:
            self.labels_ = np.zeros(n_samples, dtype=int)
            return self
        edge_nn = NearestNeighbors(n_neighbors=self.k_assign)
        edge_nn.fit(edge_objects)
        _, edge_neighbor_indices = edge_nn.kneighbors(edge_objects)
        subclusters = []
        for i in range(len(edge_indices)):
            cluster = set(edge_indices[edge_neighbor_indices[i]])
            subclusters.append(cluster)
        merged = True
        while merged:
            merged = False
            for i in range(len(subclusters)):
                for j in range(i + 1, len(subclusters)):
                    if subclusters[i].intersection(subclusters[j]):
                        subclusters[i] = subclusters[i].union(subclusters[j])
                        subclusters.pop(j)
                        merged = True
                        break
                if merged:
                    break
        edge_labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(subclusters):
            edge_labels[list(cluster)] = cluster_id + 1
        non_edge_indices = np.where(~edge_mask)[0]
        labels = np.zeros(n_samples, dtype=int)
        labels[edge_indices] = edge_labels[edge_indices]
        for idx in non_edge_indices:
            obj = X[idx]
            min_dist = np.inf
            best_cluster = -1
            for cluster_id, cluster in enumerate(subclusters):
                cluster_objects = X[list(cluster)]
                dists = np.linalg.norm(cluster_objects - obj, axis=1)
                current_min = np.min(dists)
                if current_min < min_dist:
                    min_dist = current_min
                    best_cluster = cluster_id + 1
            labels[idx] = best_cluster
        self.labels_ = labels - 1  # 转换为0 - based索引
        return self

if __name__ == "__main__":
    yeast = fetch_ucirepo(id=45)
    X = yeast.data.features.values  
    y = yeast.data.targets.values  
    mask = ~np.isnan(X).any(axis=1)  
    X = X[mask]
    y = y[mask] 
    # Max - Min 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

 
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    le = LabelEncoder()
    true_labels = le.fit_transform(y.ravel())
    k_gene = 8 
    k_subcluster = 2  
    radio_percent=75
    gfc = GFC(k_gene_force=k_gene, k_assign=k_subcluster,radio=radio_percent)
    gfc.fit(X_pca)
    cluster_labels = gfc.labels_
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    print(f"自动计算的theta值: {gfc.theta:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")