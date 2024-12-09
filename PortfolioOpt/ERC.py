import numpy as np

class ERCSolver:
    """
    Classe pour résoudre le problème Equal Risk Contribution (ERC)
    via méthode itérative.
    """
    def __init__(self, covariance_matrix, tol=1e-5):
        """
        Initialise la classe ERCSolver.

        Parameters:
        - covariance_matrix (array): Matrice de covariance des rendements.
        - tol (float): Seuil de convergence pour l'algorithme.
        """
        self.Sigma = np.array(covariance_matrix)
        self.tol = tol
        self.n = self.Sigma.shape[0]  # Nombre d'actifs
        self.weights = None

    def _risk_contributions(self, x):
        """
        Calcule les contributions au risque des actifs pour un vecteur x donné.

        Parameters:
        - x (array): Poids du portefeuille.

        Returns:
        - rc (array): Contributions au risque de chaque actif.
        """
        Sx = self.Sigma.dot(x)
        portfolio_risk = np.sqrt(x.T @ Sx)
        return x * Sx / portfolio_risk

    def solve(self):
        """
        Résout le problème ERC et retourne les poids optimaux.

        Returns:
        - weights (array): Poids optimaux pour le portefeuille ERC.
        """
        # Initialisation des poids
        x0 = np.ones((self.n, 1)) / self.n
        x = x0 * 10
        var = np.diag(self.Sigma)
        Sx = self.Sigma.dot(x)
        convergence = False

        # Boucle itérative pour ajuster les poids
        while not convergence:
            for i in range(self.n):
                alpha = var[i]
                beta = (Sx[i] - x[i] * var[i])[0]
                gamma_ = -1.0 / self.n

                # Mise à jour du poids x_i
                x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma_)) / (2 * alpha)
                x[i] = x_tilde
                Sx = self.Sigma.dot(x)

            # Test de convergence
            convergence = np.sum((x / np.sum(x) - x0 / np.sum(x0))**2) <= self.tol
            x0 = x.copy()

        # Normalisation des poids
        self.weights = (x / np.sum(x)).flatten()
        return self.weights

    def get_risk_contributions(self):
        """
        Retourne les contributions au risque des actifs après résolution.

        Returns:
        - rc (array): Contributions au risque.
        """
        if self.weights is None:
            raise ValueError("Vous devez d'abord appeler la méthode solve().")
        return self._risk_contributions(self.weights)
