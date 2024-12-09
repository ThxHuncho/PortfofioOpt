import numpy as np

class RiskBudgetingSolver:
    """
    Classe pour résoudre le problème de budgetisation du risque avec des budgets de risque spécifiques.
    """
    def __init__(self, covariance_matrix, risk_budgets, tol=1e-5):
        """
        Initialise la classe RiskBudgetingSolver.

        Parameters:
        - covariance_matrix (array): Matrice de covariance des rendements.
        - risk_budgets (array): Budgets de risque assignés (somme(b) = 1).
        - tol (float): Seuil de convergence pour l'algorithme.
        """
        self.Sigma = np.array(covariance_matrix)
        self.b = np.array(risk_budgets).reshape(-1, 1)
        self.tol = tol
        self.n = self.Sigma.shape[0]  # Nombre d'actifs
        self.weights = None

    def solve(self):
        """
        Résout le problème de budgetisation du risque.

        Returns:
        - weights (array): Poids optimaux du portefeuille.
        """
        # Initialisation des poids
        x0 = np.ones((self.n, 1)) / self.n
        x = x0 * 10
        var = np.diag(self.Sigma)
        Sx = self.Sigma.dot(x)
        convergence = False

        # Boucle itérative
        while not convergence:
            for i in range(self.n):
                alpha = var[i]  # Variance de l'actif i
                beta = (Sx[i] - x[i] * var[i])[0]  # Terme intermédiaire
                gamma_ = -self.b[i]  # Prend en compte le budget de risque b_i

                # Mise à jour de x_i
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
        Calcule les contributions au risque des actifs pour les poids optimaux.

        Returns:
        - rc (array): Contributions au risque.
        """
        if self.weights is None:
            raise ValueError("Vous devez d'abord appeler la méthode solve().")
        Sx = self.Sigma.dot(self.weights)
        portfolio_risk = np.sqrt(self.weights.T @ Sx)
        risk_contributions = self.weights * Sx / portfolio_risk
        return risk_contributions
