# Importation des bibliothèques nécessaires
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns
# Configuration du style pour une meilleure lisibilité
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")


class FinancialDataHandler:
    def __init__(self, tickers, benchmark_ticker, start_date, end_date, split_date):
        """
        Classe pour gérer le téléchargement, la préparation et la séparation des données financières.

        :param tickers: Liste des tickers des actions.
        :param benchmark_ticker: Ticker de l'indice de référence (ex. ^GSPC pour S&P 500).
        :param start_date: Date de début des données.
        :param end_date: Date de fin des données.
        :param split_date: Date de séparation pour les périodes in-sample et out-of-sample.
        """
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date

        self.data = None
        self.returns = None
        self.sp500_returns = None
        self.returns_in = None
        self.returns_out = None
        self.data_out = None
        self.data_in = None

    def download_data(self):
        """Télécharge les données de clôture ajustée pour les tickers donnés."""
        self.data = yf.download(self.tickers + [self.benchmark_ticker], start=self.start_date, end=self.end_date)['Close']

    def calculate_returns(self):
        """Calcule les rendements quotidiens pour les actions et l'indice de référence."""
        self.returns = self.data[self.tickers].pct_change().dropna()
        self.sp500_returns = self.data[self.benchmark_ticker].pct_change().dropna()

    def split_data(self):
        """Sépare les données de rendements en périodes in-sample et out-of-sample."""
        self.returns_in = self.returns.loc[:self.split_date]
        self.returns_out = self.returns.loc[self.split_date:]
        self.data_out = self.data.loc[self.split_date:]
        self.data_in = self.data.loc[:self.split_date]

    def get_in_sample_return(self):
        """Retourne les données in-sample."""
        return self.returns_in

    def get_Ben_return(self):
        """Retourne les données in-sample."""
        return self.sp500_returns

    def get_out_of_sample_return(self):
        """Retourne les données out-of-sample."""
        return self.returns_out

    def get_out_of_sample_data(self):
        """Retourne les données out-of-sample."""
        return self.data_out

    def get_in_sample_data(self):
        """Retourne les données out-of-sample."""
        return self.data_in

    def process(self):
        """Pipeline complet pour le téléchargement, le calcul des rendements et la séparation des périodes."""
        self.download_data()
        self.calculate_returns()
        self.split_data()
        self.get_in_sample_return()
        self.get_Ben_return()
        self.get_out_of_sample_return()
        self.get_in_sample_data()
        self.get_out_of_sample_data()

class OptimizePortfolio:
    def __init__(self, returns):
        """
        Classe pour gérer les différentes méthodes d'optimisation de portefeuille .

        :param returns: DataFrame des rendements des actifs.
        """
        self.returns = returns
        self.n = returns.shape[1]
        self.mu = returns.mean().values
        self.Sigma = returns.cov().values

    def markowitz_linear(self, lin_cost, risk_target):
        """
        Optimisation de Markowitz avec coûts de transaction.

        :param lin_cost: Coûts linéaires associés aux actifs.
        :param risk_target: Niveau cible de risque.
        :return: Série Pandas avec les poids du portefeuille.
        """
        w = cvx.Variable(self.n)
        portfolio_return = self.mu @ w
        portfolio_risk = cvx.quad_form(w, self.Sigma)
        transaction_cost = lin_cost @ cvx.abs(w)
        objective = cvx.Maximize(portfolio_return - transaction_cost)
        constraints = [cvx.sum(w) == 1,
                       portfolio_risk <= risk_target**2,
                       w >= 0]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        return pd.Series(w.value, index=self.returns.columns)

    def markowitz_max_return(self, risk_target):
        """
        Optimisation de Markowitz pour maximiser le rendement.

        :return: Série Pandas avec les poids du portefeuille.
        """
        w = cvx.Variable(self.n)
        objective = cvx.Maximize(self.mu @ w)
        portfolio_risk = cvx.quad_form(w, self.Sigma)
        constraints = [cvx.sum(w) == 1,
                       portfolio_risk <= risk_target ** 2,
                       w >= 0]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        return pd.Series(w.value, index=self.returns.columns)

    def markowitz_utility(self, risk_aversion):
        """
        Optimisation de Markowitz maximisant l'utilité.

        :param risk_aversion: Coefficient d'aversion au risque.
        :return: Série Pandas avec les poids du portefeuille.
        """
        w = cvx.Variable(self.n)
        utility = self.mu @ w - (risk_aversion * cvx.quad_form(w, self.Sigma))/ 2
        objective = cvx.Maximize(utility)
        constraints = [cvx.sum(w) == 1,
                       w >= 0]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.ECOS)
        return pd.Series(w.value, index=self.returns.columns)

    def compute_ERC(self):
        """
        Optimisation de portefeuille Equal Risk Contribution (ERC).

        :return: Numpy array des poids du portefeuille ERC.
        """
        Sigma = np.array(self.Sigma)
        n = Sigma.shape[0]
        x0 = np.ones((n, 1)) / n
        x = x0 * 10
        var = np.diag(Sigma)
        Sx = Sigma.dot(x)
        convergence = False
        while not convergence:
            for i in range(n):
                alpha = var[i]
                beta = (Sx[i] - x[i] * var[i])[0]
                gamma_ = -1.0 / n
                x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma_)) / (2 * alpha)
                x[i] = x_tilde
                Sx = Sigma.dot(x)
            convergence = np.sum((x / np.sum(x) - x0 / np.sum(x0))**2) <= 1e-5
            x0 = x.copy()
        return (x / x.sum()).flatten()

class HierarchicalRiskParity:
    def __init__(self, corr):
        """
        Classe pour gérer les calculs et visualisations liés au HRP.

        :param corr: Matrice de corrélation des actifs.
        """
        self.corr = corr

    def correl_dist(self):
        """
        Calcule la matrice de distance basée sur la corrélation.
        """
        dist = ((1 - self.corr) / 2) ** 0.5
        return dist

    def get_quasi_diag(self, link):
        """
        Tri quasi-diagonal pour ordonner les actifs.

        :param link: Résultat du clustering hiérarchique.
        :return: Liste des indices ordonnés.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def get_cluster_var(self, cov, c_items):
        """
        Calcul de la variance par cluster.

        :param cov: Matrice de covariance.
        :param c_items: Actifs du cluster.
        :return: Variance du cluster.
        """
        cov_ = cov.loc[c_items, c_items]
        ivp = 1 / np.diag(cov_)
        ivp /= ivp.sum()
        c_var = np.dot(np.dot(ivp.T, cov_), ivp)
        return c_var

    def get_rec_bipart(self, cov, sort_ix):
        """
        Allocation bipartite récursive pour HRP.

        :param cov: Matrice de covariance.
        :param sort_ix: Indices triés quasi-diagonalement.
        :return: Poids finaux des actifs.
        """
        sort_ix = [cov.columns[int(i)] for i in sort_ix]
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_var0 = self.get_cluster_var(cov, c_items[i])
                c_var1 = self.get_cluster_var(cov, c_items[i + 1])
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items[i]] *= alpha
                w[c_items[i + 1]] *= 1 - alpha
        return w

    def plot_dendrogram_with_heatmap(self):
        """
        Affiche un dendrogramme couplé à la matrice de corrélation sous forme de heatmap.
        """
        dist = self.correl_dist()
        dist_condensed = ssd.squareform(dist)
        link = sch.linkage(dist_condensed, method='single')

        fig = plt.figure(figsize=(20, 10))

        ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.8])
        dendro = sch.dendrogram(link, orientation='left', labels=self.corr.columns, ax=ax1, leaf_font_size=10)
        ax1.set_title("Dendrogramme des actifs")

        idx = dendro['leaves']
        corr_reordered = self.corr.iloc[idx, idx]

        ax2 = fig.add_axes([0.35, 0.1, 0.6, 0.8])
        cax = ax2.matshow(corr_reordered, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(cax, ax=ax2)

        ax2.set_xticks(range(len(self.corr.columns)))
        ax2.set_xticklabels(corr_reordered.columns, rotation=90, fontsize=10)
        ax2.set_yticks(range(len(self.corr.columns)))
        ax2.set_yticklabels(corr_reordered.columns, fontsize=10)
        ax2.set_title("Matrice de corrélation réorganisée")

        plt.tight_layout()
        plt.savefig("Dendrogram_Heatmap_Correlation.png")
        plt.show()

    def plot_corr_matrix(self, labels=None):
        """
        Affiche la heatmap simple de la matrice de corrélation.
        """
        if labels is None:
            labels = self.corr.columns
        plt.figure(figsize=(8, 6))
        plt.pcolor(self.corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.yticks(np.arange(0.5, self.corr.shape[0] + 0.5), labels)
        plt.xticks(np.arange(0.5, self.corr.shape[0] + 0.5), labels, rotation=90)
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self):
        """
        Affiche uniquement le dendrogramme.
        """
        dist = self.correl_dist()
        dist_condensed = ssd.squareform(dist)
        link = sch.linkage(dist_condensed, method='single')

        plt.figure(figsize=(10, 6))
        sch.dendrogram(link, labels=self.corr.columns, orientation='top', leaf_font_size=10)
        plt.title("Dendrogramme des actifs")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig("HRP_Dendrogram.png")
        plt.show()

    def plot_reordered_heatmap(self):
        """
        Affiche uniquement la matrice de corrélation réorganisée.
        """
        dist = self.correl_dist()
        dist_condensed = ssd.squareform(dist)
        link = sch.linkage(dist_condensed, method='single')
        dendro = sch.dendrogram(link, no_plot=True)

        idx = dendro['leaves']
        corr_reordered = self.corr.iloc[idx, idx]

        plt.figure(figsize=(20, 10))
        plt.matshow(corr_reordered, cmap='coolwarm', vmin=-1, vmax=1, fignum=1)
        plt.colorbar()
        plt.title("Matrice de corrélation réorganisée", pad=20)
        plt.xticks(range(len(self.corr.columns)), corr_reordered.columns, rotation=90)
        plt.yticks(range(len(self.corr.columns)), corr_reordered.columns)
        plt.tight_layout()
        plt.savefig("HRP_Reordered_Heatmap.png")
        plt.show()


class Backtester:
    def __init__(self, returns, optimization_method):
        """
        Classe pour effectuer des backtests avec fenêtres glissantes.

        :param returns: DataFrame des rendements des actifs.
        :param optimization_method: Méthode d'optimisation de portefeuille (doit être une méthode de OptimizePortfolio).
        """
        self.returns = returns
        self.optimization_method = optimization_method

    def run_backtest(self, start_year, window_size, step_size):
        """
        Effectue le backtest avec des fenêtres glissantes dynamiques basées sur des années.

        :param start_year: Première année du backtest.
        :param window_size: Taille de la fenêtre in-sample (en années).
        :param step_size: Taille du pas pour le out-of-sample (en années).
        :return: DataFrame des poids et Série de performances cumulées.
        """
        weights = []
        combined_performance = []  # Liste pour stocker les séries de performances
        returns = self.returns.copy()

        # Initialisation de la première date
        start_date = pd.to_datetime(f"{start_year}-01-01")

        while start_date + pd.DateOffset(years=window_size + step_size) <= returns.index[-1]:
            # Définir les périodes in-sample et out-of-sample
            in_sample_end = start_date + pd.DateOffset(years=window_size - 1, months=11, days=30)
            out_sample_start = in_sample_end + pd.DateOffset(days=1)
            out_sample_end = out_sample_start + pd.DateOffset(years=step_size - 1, months=11, days=30)

            # Extraire les données pour in-sample et out-of-sample
            in_sample_returns = returns.loc[start_date:in_sample_end]
            out_sample_returns = returns.loc[out_sample_start:out_sample_end]

            # Optimiser le portefeuille sur la période in-sample
            optimizer = OptimizePortfolio(in_sample_returns)
            weight = self.optimization_method(optimizer)
            weights.append(weight)

            # Calculer les performances out-of-sample
            performance = (out_sample_returns * weight).sum(axis=1)
            combined_performance.append(performance)

            # Avancer la fenêtre
            start_date += pd.DateOffset(years=step_size)

        # Créer un DataFrame pour les poids
        weights_df = pd.DataFrame(weights, index=pd.date_range(start=f"{start_year + window_size}-01-01", periods=len(weights), freq='YS'))
        combined_performance = pd.concat(combined_performance).cumsum()  # Combiner toutes les performances

        return weights_df, combined_performance

    def plot_performance(self, combined_performance):
        """
        Trace la performance cumulée.

        :param combined_performance: Série de performance cumulée.
        """
        plt.figure(figsize=(12, 8))
        plt.plot(combined_performance, label="Performance cumulée", linewidth=2)
        plt.title("Performance cumulée avec fenêtres glissantes dynamiques")
        plt.xlabel("Date")
        plt.ylabel("Performance cumulée")
        plt.legend()
        plt.grid(True)
        plt.show()


def performance_stats(returns):
    """
    Calcule les statistiques de performance :
    - Mean Return annualisé
    - Volatilité annualisée
    - Ratio de Sharpe
    - Cumulative Return
    """
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = mean_return / volatility
    cumulative_return = returns.sum()
    return {"Mean Return": np.round(mean_return,2),
            "Volatility": np.round(volatility,2),
            "Sharpe Ratio": np.round(sharpe_ratio,2),
            "Cumulative Return": np.round(cumulative_return,2)}

def TrendFollowingStrat(data, tickers, weights_linear, weights_utility, weights_hrp, weights_erc):
    """
    Implémente une stratégie de suivi de tendance avec différentes méthodes, y compris Equal Weight (EW).

    Args:
        data (pd.DataFrame): Données de prix avec une colonne 'Date' et des colonnes par ticker.
        tickers (list): Liste des tickers.
        weights_linear (dict): Poids pour la stratégie "Linear".
        weights_utility (dict): Poids pour la stratégie "Utility".
        weights_hrp (dict): Poids pour la stratégie "HRP".
        weights_erc (list): Poids pour la stratégie "ERC".

    Returns:
        allPricesClean (pd.DataFrame): Données nettoyées avec signaux et poids.
        portRes (pd.DataFrame): Résultats des stratégies avec rendements cumulés.
    """
    # Mise en forme des données pour le traitement
    allPrices = data.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="c")
    allPrices['NextOpen'] = allPrices.groupby('Ticker')['c'].shift(-1)

    # Calcul des rendements logarithmiques avec transform
    allPrices['Return'] = allPrices.groupby('Ticker')['c'].transform(lambda x: np.log(x).diff())
    allPrices['ReturnTC'] = allPrices.groupby('Ticker')['NextOpen'].transform(lambda x: np.log(x).diff())

    # Calcul de la volatilité roulante (256 jours, écart-type)
    rolling_window = 256
    allPrices['RunVol'] = allPrices.groupby('Ticker')['Return'].transform(lambda x: x.rolling(rolling_window).std())

    # Normalisation des rendements
    scaling_factor = 0.1
    allPrices['rhat'] = allPrices['Return'] * scaling_factor / allPrices['RunVol']

    # Suppression des valeurs NaN issues des calculs roulants
    allPrices.dropna(subset=['Return', 'RunVol', 'rhat'], inplace=True)

    # Nettoyage des données
    allPricesClean = allPrices.dropna(subset=['rhat']).copy()

    # Calcul des sommes cumulées groupées par 'Ticker'
    allPricesClean['rhatC'] = allPricesClean.groupby('Ticker')['rhat'].cumsum()
    allPricesClean['rc'] = allPricesClean.groupby('Ticker')['Return'].cumsum()

    # Calcul de moyennes mobiles et des signaux
    allPricesClean['RunMean'] = allPricesClean.groupby('Ticker')['rhat'].transform(lambda x: x.rolling(10).mean())
    allPricesClean['Signal'] = np.sign(allPricesClean['RunMean'])
    allPricesClean['SignalLO'] = allPricesClean['RunMean'] > 0
    allPricesClean['SignalShifted'] = allPricesClean.groupby('Ticker')['Signal'].shift(1)
    allPricesClean['SignalLOShifted'] = allPricesClean.groupby('Ticker')['SignalLO'].shift(1)

    # Ajout des poids pour chaque stratégie
    allPricesClean['Poids Trans cost'] = allPricesClean['Ticker'].map(weights_linear)
    allPricesClean['Poids Utility'] = allPricesClean['Ticker'].map(weights_utility)
    allPricesClean['Poids HRP'] = allPricesClean['Ticker'].map(weights_hrp)
    allPricesClean['Poids ERC'] = allPricesClean['Ticker'].map(dict(zip(tickers, weights_erc)))

    # Fonction pour calculer les rendements totaux, y compris EW
    def calculate_total_returns(group):
        return pd.Series({
            # TransCost
            'TotalReturnLO TransCost': (group['SignalLOShifted'] * group['Return'] * group['Poids Trans cost']).sum(),
            # Utility
            'TotalReturnLO Utility': (group['SignalLOShifted'] * group['Return'] * group['Poids Utility']).sum(),
            # HRP
            'TotalReturnLO HRP': (group['SignalLOShifted'] * group['Return'] * group['Poids HRP']).sum(),
            # ERC
            'TotalReturnLO ERC': (group['SignalLOShifted'] * group['Return'] * group['Poids ERC']).sum(),
            # Equal Weight (EW)
            'TotalReturnLO EW': (1 / len(tickers)) * (group['SignalLOShifted'] * group['Return']).sum(),
        })

    # Application de la fonction par groupe (Date)
    portRes = allPricesClean.groupby('Date').apply(calculate_total_returns).reset_index()

    # Ajout de rendements cumulés pour chaque stratégie
    strategies = ['TotalReturnLO TransCost',
                  'TotalReturnLO Utility',
                  'TotalReturnLO HRP',
                  'TotalReturnLO ERC',
                  'TotalReturnLO EW']

    for strategy in strategies:
        portRes[f'Cumulative {strategy}'] = (1 + portRes[strategy]).cumprod() - 1

    stats = []
    for strategy in strategies:
        returns = portRes[strategy]
        cumulative_return = portRes[f'Cumulative {strategy}'].iloc[-1]
        mean_return = returns.mean() * 252  # Annualized mean return (assuming 252 trading days)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (mean_return ) / volatility if volatility != 0 else np.nan

        stats.append({
            'Strategy': strategy,
            'Cumulative Return': np.round(cumulative_return,2),
            'Annualized Mean Return': np.round(mean_return,2),
            'Annualized Volatility': np.round(volatility,2),
            'Sharpe Ratio': np.round(sharpe_ratio,2)
        })

    stats_df = pd.DataFrame(stats)

    return allPricesClean, portRes, stats_df


def TrendFollowingStratL1(data, ReturnInfo, tickers, weights_linear, weights_utility, weights_hrp, weights_erc):
    """
    Implémente une stratégie de suivi de tendance basée sur un filtre L1.

    Args:
        data (pd.DataFrame): Données de prix avec une colonne 'Date' et des colonnes par ticker.
        ReturnInfo (pd.DataFrame): Données supplémentaires avec rendements pré-calculés (rhat, Return, ReturnTC, rhatC, rc).
        tickers (list): Liste des tickers.
        weights_linear (dict): Poids pour la stratégie "Linear".
        weights_utility (dict): Poids pour la stratégie "Utility".
        weights_hrp (dict): Poids pour la stratégie "HRP".
        weights_erc (list): Poids pour la stratégie "ERC".

    Returns:
        allPricesClean (pd.DataFrame): Données nettoyées avec signaux et poids.
        portRes (pd.DataFrame): Résultats des stratégies avec rendements cumulés.
    """
    # Mise en forme des données
    allPrices = data.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="c")
    allPrices['NextOpen'] = allPrices.groupby('Ticker')['c'].shift(-1)

    # Calcul des rendements logarithmiques
    allPrices['Return'] = allPrices.groupby('Ticker')['c'].transform(lambda x: np.log(x).diff())
    allPrices['ReturnTC'] = allPrices.groupby('Ticker')['NextOpen'].transform(lambda x: np.log(x).diff())

    # Calcul de la volatilité roulante (256 jours, écart-type)
    rolling_window = 256
    allPrices['RunVol'] = allPrices.groupby('Ticker')['Return'].transform(lambda x: x.rolling(rolling_window).std())

    # Normalisation des rendements
    scaling_factor = 0.1
    allPrices['rhat'] = allPrices['Return'] * scaling_factor / allPrices['RunVol']

    # Suppression des NaN
    allPrices.dropna(subset=['Return', 'RunVol', 'rhat'], inplace=True)

    # Nettoyage des données
    allPricesClean = allPrices.dropna(subset=['rhat']).copy()

    # Ajout des informations de ReturnInfo
    for col in ['rhatC', 'rc', 'rhat', 'Return', 'ReturnTC']:
        allPricesClean[f"{col} f"] = ReturnInfo[col]

    # Calcul des moyennes mobiles et des signaux
    allPricesClean['RunMean'] = allPricesClean.groupby('Ticker')['rhat'].transform(lambda x: x.rolling(10).mean())
    allPricesClean['Signal'] = np.sign(allPricesClean['RunMean'])
    allPricesClean['SignalLO'] = allPricesClean['RunMean'] > 0
    allPricesClean['SignalShifted'] = allPricesClean.groupby('Ticker')['Signal'].shift(1)
    allPricesClean['SignalLOShifted'] = allPricesClean.groupby('Ticker')['SignalLO'].shift(1)

    # Ajout des poids pour chaque stratégie
    allPricesClean['Poids Trans cost'] = allPricesClean['Ticker'].map(weights_linear)
    allPricesClean['Poids Utility'] = allPricesClean['Ticker'].map(weights_utility)
    allPricesClean['Poids HRP'] = allPricesClean['Ticker'].map(weights_hrp)
    allPricesClean['Poids ERC'] = allPricesClean['Ticker'].map(dict(zip(tickers, weights_erc)))

    # Calcul des rendements totaux, y compris EW
    def calculate_total_returns(group):
        return pd.Series({
            # TransCost
            'TotalReturnLO TransCost': (group['SignalLOShifted'] * group['Return f'] * group['Poids Trans cost']).sum(),
            # Utility
            'TotalReturnLO Utility': (group['SignalLOShifted'] * group['Return f'] * group['Poids Utility']).sum(),
            # HRP
            'TotalReturnLO HRP': (group['SignalLOShifted'] * group['Return f'] * group['Poids HRP']).sum(),
            # ERC
            'TotalReturnLO ERC': (group['SignalLOShifted'] * group['Return f'] * group['Poids ERC']).sum(),
            # Equal Weight (EW)
            'TotalReturnLO EW': (1 / len(tickers)) * (group['SignalLOShifted'] * group['Return f']).sum(),
        })

    # Application de la fonction par groupe (Date)
    portRes = allPricesClean.groupby('Date').apply(calculate_total_returns).reset_index()

    # Ajout de rendements cumulés pour chaque stratégie
    strategies = ['TotalReturnLO TransCost',
                  'TotalReturnLO Utility',
                  'TotalReturnLO HRP',
                  'TotalReturnLO ERC',
                  'TotalReturnLO EW']

    for strategy in strategies:
        portRes[f'Cumulative {strategy}'] = (1 + portRes[strategy]).cumprod() - 1

    stats = []
    for strategy in strategies:
        returns = portRes[strategy]
        cumulative_return = portRes[f'Cumulative {strategy}'].iloc[-1]
        mean_return = returns.mean() * 252  # Annualized mean return (assuming 252 trading days)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (mean_return ) / volatility if volatility != 0 else np.nan

        stats.append({
            'Strategy': strategy,
            'Cumulative Return': np.round(cumulative_return,2),
            'Annualized Mean Return': np.round(mean_return,2),
            'Annualized Volatility': np.round(volatility,2),
            'Sharpe Ratio': np.round(sharpe_ratio,2)
        })

    stats_df = pd.DataFrame(stats)

    return allPricesClean, portRes, stats_df


def plot_financial_data(portRes):
    """
    Visualisation des performances cumulées pour différentes stratégies.

    Args:
        portRes (pd.DataFrame): Résultats avec rendements cumulés pour chaque stratégie.
    """
    plt.figure(figsize=(12, 6))  # Taille de la figure

    # Tracer les courbes pour différentes stratégies
    plt.plot(portRes['Date'], 1 + portRes['Cumulative TotalReturnLO TransCost'], label='TransCost', linewidth=2)
    plt.plot(portRes['Date'], 1 + portRes['Cumulative TotalReturnLO Utility'], label='Utility', linewidth=2)
    plt.plot(portRes['Date'], 1 + portRes['Cumulative TotalReturnLO HRP'], label='HRP', linewidth=2)
    plt.plot(portRes['Date'], 1 + portRes['Cumulative TotalReturnLO ERC'], label='ERC', linewidth=2)

    # Ajouter des étiquettes et une légende
    plt.title("Performances Out-of-Sample of Trend Following", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Rendement cumulé", fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45, ha='right')  # Rotation des étiquettes pour les dates
    plt.margins(x=0.05, y=0.1)  # Marges pour éviter d'être collé aux axes
    plt.grid(False, linestyle='--', alpha=0.7)  # Ajouter une grille pour améliorer la lisibilité

    # Sauvegarder le graphique
    # plt.savefig("performances_out_of_sample_trend_following.png", dpi=300, bbox_inches='tight')

    # Afficher le graphique
    plt.show()
