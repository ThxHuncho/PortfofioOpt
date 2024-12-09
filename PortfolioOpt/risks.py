
import numpy as np

class Risks:
    @staticmethod
    def variance(weights, covariance_matrix):
        return weights.T @ covariance_matrix @ weights

    @staticmethod
    def volatility(weights, covariance_matrix):
        return np.sqrt(Risks.variance(weights, covariance_matrix))
