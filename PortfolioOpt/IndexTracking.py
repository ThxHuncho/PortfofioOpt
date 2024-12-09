import numpy as np
import cvxpy as cp

class IndexTracking:
    def __init__(self, Sigma, mu, b, phi=0.0, l1_lambda=0.0):
        """
        Initialise la classe pour l'optimisation de gestion indicielle.

        Parameters:
        - Sigma (array): Matrice de covariance des rendements.
        - mu (array): Rendements moyens des actifs.
        - b (array): Poids du benchmark.
        - phi (float): Paramètre d'aversion à la tracking error.
        - l1_lambda (float): Paramètre de régularisation L1 pour sparsité.
        """
        self.Sigma = Sigma
        self.mu = mu
        self.b = b
        self.phi = phi
        self.l1_lambda = l1_lambda
        self.weights = None

    def solve(self):
        """
        Résout le problème d'optimisation pour battre ou répliquer un benchmark.

        Returns:
        - weights (array): Poids optimaux.
        """
        n = len(self.b)
        x = cp.Variable(n, nonneg=True)

        # Fonction objective : Minimisation TE + Surperformance + L1
        tracking_error = cp.quad_form(x - self.b, self.Sigma)
        excess_return = self.phi * self.mu.T @ (x - self.b)
        l1_norm = self.l1_lambda * cp.norm1(x)

        objective = cp.Minimize(0.5 * tracking_error - excess_return + l1_norm)

        # Contraintes
        constraints = [cp.sum(x) == 1]

        # Résolution
        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.weights = x.value
        return self.weights
