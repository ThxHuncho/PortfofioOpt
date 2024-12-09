import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    def __init__(self, returns):
        """
        Initialise la classe HRP.

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
        Applique un clustering hiérarchique à la matrice de distance.
        """
        self.compute_distances()
        self.linkage_matrix = linkage(squareform(self.distances), method='single')
        return self.linkage_matrix

    def get_quasi_diagonal(self, linkage_matrix):
        """
        Ordonnancement des actifs selon la structure hiérarchique.
        """
        sort_order = [linkage_matrix[-1, 0], linkage_matrix[-1, 1]]
        sort_order = [int(i) for i in sort_order if i < len(self.correlation)]
        return sort_order

    def recursive_bisection(self, sort_order):
        """
        Allocation par Risk Parity selon la structure hiérarchique.
        """
        weights = pd.Series(1, index=sort_order)
        cluster = [sort_order]

        while len(cluster) > 0:
            cluster = [i for sub in cluster for i in (sub[:len(sub)//2], sub[len(sub)//2:]) if len(sub) > 1]
            for sub in cluster:
                left = self.covariance[np.ix_(sub, sub)].sum()
                right = self.covariance[np.ix_(sort_order, sort_order)].sum()
                weights.loc[sub] *= left / (left + right)
        return weights / weights.sum()

    def solve(self):
        """
        Résout le problème HRP et retourne les poids optimaux.
        """
        linkage_matrix = self.hierarchical_clustering()
        sort_order = self.get_quasi_diagonal(linkage_matrix)
        self.weights = self.recursive_bisection(sort_order)
        return self.weights
