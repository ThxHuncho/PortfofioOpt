import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA

class CovarianceEstimator:
    def __init__(self, returns):
        """
        Initialise la classe pour estimer la matrice de covariance.

        Parameters:
        - returns (DataFrame): Rendements historiques des actifs.
        """
        self.returns = returns
        self.cov_matrix = None  # Pour stocker la matrice de covariance

    def empirical(self):
        """
        Calcule la matrice de covariance empirique (historique).
        """
        self.cov_matrix = self.returns.cov() * 252  # Annualisée
        return self.cov_matrix

    def ledoit_wolf(self):
        """
        Calcule la matrice de covariance avec le shrinkage de Ledoit-Wolf.
        """
        lw = LedoitWolf()
        lw.fit(self.returns)
        self.cov_matrix = lw.covariance_ * 252  # Annualisée
        return pd.DataFrame(self.cov_matrix, index=self.returns.columns, columns=self.returns.columns)

    def pca_factorial(self, n_factors=2):
        """
        Calcule la matrice de covariance factorielle basée sur PCA.

        Parameters:
        - n_factors (int): Nombre de facteurs principaux à utiliser.
        """
        pca = PCA(n_components=n_factors)
        pca.fit(self.returns)
        factor_loadings = pca.components_.T
        factor_cov = np.diag(pca.explained_variance_)
        specific_variance = np.diag(np.var(self.returns - self.returns @ factor_loadings, axis=0))

        # Reconstruction de la matrice de covariance
        self.cov_matrix = factor_loadings @ factor_cov @ factor_loadings.T + specific_variance
        return pd.DataFrame(self.cov_matrix, index=self.returns.columns, columns=self.returns.columns)

    def robust_mcd(self):
        """
        Calcule la matrice de covariance robuste avec Minimum Covariance Determinant (MCD).
        """
        mcd = MinCovDet()
        mcd.fit(self.returns)
        self.cov_matrix = mcd.covariance_ * 252  # Annualisée
        return pd.DataFrame(self.cov_matrix, index=self.returns.columns, columns=self.returns.columns)

    def get_covariance(self):
        """
        Retourne la matrice de covariance calculée.
        """
        if self.cov_matrix is not None:
            return self.cov_matrix
        else:
            raise ValueError("Aucune matrice de covariance n'a été calculée.")
