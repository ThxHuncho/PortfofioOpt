
import numpy as np

class Optimization:
    def __init__(self, returns):
        self.returns = returns
        self.n_assets = returns.shape[1]

    def equal_weighted(self):
        weights = np.ones(self.n_assets) / self.n_assets
        return weights

    def inverse_volatility(self):
        volatilities = np.std(self.returns, axis=0)
        weights = 1 / volatilities
        weights /= weights.sum()
        return weights

    def random_allocation(self, seed=None):
        if seed:
            np.random.seed(seed)
        weights = np.random.dirichlet(np.ones(self.n_assets))
        return weights
