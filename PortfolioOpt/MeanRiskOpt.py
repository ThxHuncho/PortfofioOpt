import numpy as np
import cvxpy as cp

class PortfolioOptimization:
    def __init__(self, expected_returns, covariance_matrix):
        """
        Initialise la classe d'optimisation de portefeuille.

        Parameters:
        - expected_returns (array): Rendements moyens des actifs.
        - covariance_matrix (array): Matrice de covariance des rendements.
        """
        self.mu = expected_returns
        self.Sigma = covariance_matrix
        self.N = len(expected_returns)  # Nombre d'actifs

    def minimize_risk(self):
        """
        Minimiser le risque du portefeuille.
        """
        w = cp.Variable(self.N)
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value

    def maximize_expected_return(self):
        """
        Maximiser le rendement attendu.
        """
        w = cp.Variable(self.N)
        objective = cp.Maximize(self.mu @ w)
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value

    def maximize_utility(self, risk_aversion):
        """
        Maximiser l'utilité (rendement pondéré par le risque).

        Parameters:
        - risk_aversion (float): Coefficient d'aversion au risque (lambda).
        """
        w = cp.Variable(self.N)
        objective = cp.Maximize(self.mu @ w - risk_aversion / 2 * cp.quad_form(w, self.Sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value

    def maximize_sharpe_ratio(self):
        """
        Maximiser le ratio de Sharpe.
        """
        w = cp.Variable(self.N)
        objective = cp.Maximize((self.mu @ w) / cp.sqrt(cp.quad_form(w, self.Sigma)))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value
