import numpy as np
import cvxpy as cp

class Optimization:
    def __init__(self, returns, covariance_matrix=None):
        """
        Initialize the Optimization class with returns and optionally a covariance matrix.

        :param returns: np.ndarray
            Matrix of historical returns (rows: time, columns: assets).
        :param covariance_matrix: np.ndarray, optional
            Precomputed covariance matrix (default: computed from returns).
        """
        self.returns = returns
        self.mean_returns = np.mean(returns, axis=0)
        self.covariance_matrix = covariance_matrix or np.cov(returns, rowvar=False)
        self.n_assets = returns.shape[1]

    def mean_variance(self, risk_aversion=1.0):
        """
        Mean-variance optimization.

        :param risk_aversion: float
            Risk aversion parameter (higher values penalize risk more heavily).
        :return: np.ndarray
            Optimal weights.
        """
        weights = cp.Variable(self.n_assets)
        objective = cp.quad_form(weights, self.covariance_matrix) - risk_aversion * self.mean_returns @ weights
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return weights.value

    def sharpe_ratio(self, risk_free_rate=0.0):
        """
        Maximize the Sharpe ratio reformulated to avoid non-DCP constraints.
    
        :param risk_free_rate: float
            Risk-free rate (default: 0).
        :return: np.ndarray
            Optimal weights.
        """
        weights = cp.Variable(self.n_assets)
    
        # Reformulated Sharpe ratio objective
        portfolio_return = self.mean_returns @ weights - risk_free_rate
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
    
        # Objective: Maximize portfolio return divided by portfolio variance
        objective = cp.Maximize(portfolio_return / portfolio_variance)
    
        # Constraints
        constraints = [cp.sum(weights) == 1, weights >= 0]
    
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
    
        return weights.value

    def risk_parity(self):
        """
        Equal Risk Contribution (ERC).

        :return: np.ndarray
            Optimal weights.
        """
        weights = cp.Variable(self.n_assets)
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
        marginal_risk = self.covariance_matrix @ weights
        risk_contribution = cp.multiply(weights, marginal_risk) / cp.sqrt(portfolio_variance)
        objective = cp.norm(risk_contribution - cp.sum(risk_contribution) / self.n_assets)
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return weights.value

    def maximum_diversification(self):
        """
        Maximum diversification optimization.

        :return: np.ndarray
            Optimal weights.
        """
        volatilities = np.sqrt(np.diag(self.covariance_matrix))
        weights = cp.Variable(self.n_assets)
        diversification_ratio = cp.sum(cp.multiply(weights, volatilities)) / cp.sqrt(cp.quad_form(weights, self.covariance_matrix))
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Maximize(diversification_ratio), constraints)
        problem.solve()
        return weights.value

    def robust_cvar(self, alpha=0.95):
        """
        Distributionally Robust CVaR optimization.

        :param alpha: float
            Confidence level for CVaR (default: 0.95).
        :return: np.ndarray
            Optimal weights.
        """
        weights = cp.Variable(self.n_assets)
        z = cp.Variable(len(self.returns))
        t = cp.Variable()
        portfolio_returns = self.returns @ weights
        cvar_constraint = portfolio_returns + z >= t
        objective = t + (1 / (1 - alpha)) * cp.sum(z)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            z >= 0,
            cvar_constraint
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return weights.value
