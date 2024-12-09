import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalEqualRiskContribution:
    def __init__(self, returns):
        """
        Initialise la classe HERC.

        Parameters:
        - returns (DataFrame): Rendements historiques des actifs.
        """
        self.returns = returns
        self.correlation = returns.corr().values
        self.covariance = returns.cov().values
        self.distances = None
        self.linkage_matrix = None
        self.weights = None

    def compute_distances(self):
        """
        Calcule la matrice de distances pour le clustering.
        """
        self.distances = np.sqrt(2 * (1 - self.correlation))

    def hierarchical_clustering(self):
        """
        Applique le clustering hiérarchique pour organiser les actifs.
        """
        self.compute_distances()
        self.linkage_matrix = linkage(squareform(self.distances), method='ward')
        return self.linkage_matrix

    def quasi_diagonal_sort(self, linkage_matrix):
        """
        Renvoie l'ordre quasi-diagonal des actifs après clustering.
        """
        sort_order = []
        def _sort(tree):
            if tree < len(self.correlation):
                sort_order.append(tree)
            else:
                left = int(linkage_matrix[tree - len(self.correlation), 0])
                right = int(linkage_matrix[tree - len(self.correlation), 1])
                _sort(left)
                _sort(right)
        _sort(len(linkage_matrix) + len(self.correlation) - 2)
        return sort_order

    def calculate_erc(self, cluster_indices):
        """
        Calcule les poids selon la méthode Equal Risk Contribution pour un cluster donné.
        """
        sub_cov = self.covariance[np.ix_(cluster_indices, cluster_indices)]
        n = len(cluster_indices)
        weights = np.ones(n) / n
        for _ in range(100):  # Boucle de convergence
            marginal_risk = sub_cov @ weights
            risk_contributions = weights * marginal_risk
            total_risk = np.sum(risk_contributions)
            weights *= total_risk / risk_contributions
            weights /= np.sum(weights)
        return weights

    def solve(self):
        """
        Résout le problème HERC et retourne les poids optimaux.
        """
        linkage_matrix = self.hierarchical_clustering()
        sort_order = self.quasi_diagonal_sort(linkage_matrix)

        # Divise les actifs en clusters et applique ERC récursivement
        cluster_indices = [sort_order]
        weights = pd.Series(1, index=sort_order)

        while len(cluster_indices) > 0:
            new_clusters = []
            for indices in cluster_indices:
                if len(indices) > 1:
                    half = len(indices) // 2
                    left = indices[:half]
                    right = indices[half:]
                    new_clusters.append(left)
                    new_clusters.append(right)

                    # Applique ERC entre les deux sous-clusters
                    w_left = np.sum(weights[left])
                    w_right = np.sum(weights[right])
                    total_weight = w_left + w_right
                    weights[left] *= total_weight / 2 / w_left
                    weights[right] *= total_weight / 2 / w_right
            cluster_indices = new_clusters

        # Normalise les poids
        self.weights = weights / np.sum(weights)
        return self.weights
