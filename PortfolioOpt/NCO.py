import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import LedoitWolf

class NestedClustersOptimization:
    def __init__(self, returns, n_clusters=2):
        """
        Initialise la classe pour Nested Clusters Optimization (NCO).

        Parameters:
        - returns (DataFrame): Rendements historiques des actifs.
        - n_clusters (int): Nombre de clusters à optimiser.
        """
        self.returns = returns
        self.n_clusters = n_clusters
        self.cov_matrix = None
        self.clusters = None
        self.weights = None

    def estimate_covariance(self):
        """
        Estime la matrice de covariance en utilisant la méthode de Ledoit-Wolf pour plus de robustesse.
        """
        lw = LedoitWolf()
        self.cov_matrix = lw.fit(self.returns).covariance_

    def perform_clustering(self):
        """
        Effectue un clustering hiérarchique basé sur la matrice de corrélation.
        """
        corr_matrix = np.corrcoef(self.returns, rowvar=False)
        distances = np.sqrt(2 * (1 - corr_matrix))
        linkage_matrix = linkage(distances, method='ward')
        self.clusters = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')

    def optimize_cluster(self, assets):
        """
        Optimise un cluster localement en minimisant la variance.
        """
        cov_sub = self.cov_matrix[np.ix_(assets, assets)]
        inv_cov = np.linalg.pinv(cov_sub)  # Inverse pseudo-généralisée pour stabilité
        ones = np.ones(len(assets))
        weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
        return weights

    def solve(self):
        """
        Résout le problème NCO : clustering -> optimisation locale -> combinaison des clusters.
        """
        self.estimate_covariance()
        self.perform_clustering()
        unique_clusters = np.unique(self.clusters)

        cluster_weights = {}
        for cluster in unique_clusters:
            assets = np.where(self.clusters == cluster)[0]
            cluster_weights[cluster] = self.optimize_cluster(assets)

        # Combiner les clusters en portefeuille global
        global_weights = np.zeros(self.returns.shape[1])
        for cluster, weights in cluster_weights.items():
            assets = np.where(self.clusters == cluster)[0]
            global_weights[assets] = weights / len(unique_clusters)

        self.weights = global_weights
        return self.weights
