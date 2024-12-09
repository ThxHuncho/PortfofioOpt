import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

class ExpectedReturnsEstimator:
    def __init__(self, price_data, risk_free_rate=0.02, market_returns=None):
        """
        Initialise la classe pour estimer les rendements attendus.
        
        Parameters:
        - price_data (DataFrame): Prix historiques des actifs (colonnes = actifs, lignes = dates).
        - risk_free_rate (float): Taux sans risque annuel (par défaut 2%).
        - market_returns (Series): Rendements journaliers du marché pour le modèle CAPM.
        """
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate
        self.market_returns = market_returns
        self.log_returns = np.log(price_data / price_data.shift(1)).dropna()
        self.annualization_factor = 252  # Nombre de jours de trading par an

    def mean_historical(self):
        """
        Calcule les rendements attendus via la moyenne historique annualisée.
        """
        return self.log_returns.mean() * self.annualization_factor

    def capm(self):
        """
        Calcule les rendements attendus via le modèle CAPM.
        """
        if self.market_returns is None:
            raise ValueError("Les rendements du marché sont requis pour CAPM.")

        betas = []
        for col in self.log_returns.columns:
            X = self.market_returns.values.reshape(-1, 1)
            y = self.log_returns[col].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0][0]
            betas.append(beta)
        
        # Calcul des rendements attendus avec CAPM
        market_premium = self.market_returns.mean() * self.annualization_factor - self.risk_free_rate
        expected_returns_capm = self.risk_free_rate + np.array(betas) * market_premium
        
        return pd.Series(expected_returns_capm, index=self.log_returns.columns)

    def fama_french(self, factor_data):
        """
        Calcule les rendements attendus en utilisant un modèle multifactoriel (ex : Fama-French 3 facteurs).
        
        Parameters:
        - factor_data (DataFrame): Facteurs comme SMB, HML et MKT.
        """
        expected_returns = []
        for col in self.log_returns.columns:
            X = factor_data.values
            y = self.log_returns[col].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            factor_means = factor_data.mean().values
            expected_return = intercept + np.sum(coefficients * factor_means)
            expected_returns.append(expected_return * self.annualization_factor)
        
        return pd.Series(expected_returns, index=self.log_returns.columns)
