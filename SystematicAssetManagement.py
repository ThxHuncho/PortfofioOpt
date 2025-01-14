from rich.jupyter import display

from SystematicAM import *

if __name__ == "__main__":

    #tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL","GOOG", "WMT", "UNH"]
    tickers = ["XOM", "MSFT", "C", "GE", "WMT", "BAC", "JNJ", "PFE", "INTC", "AIG"]

    benchmark_ticker = '^GSPC'
    start_date = "2005-01-01"
    end_date = "2025-01-01"
    split_date = "2015-01-01"

    handler = FinancialDataHandler(tickers, benchmark_ticker, start_date, end_date, split_date)
    handler.process()

    # Récupérer les données in-sample et out-of-sample
    returns_in = handler.get_in_sample_return()
    returns_out = handler.get_out_of_sample_return()
    data_out = handler.get_out_of_sample_data()
    data_in = handler.get_in_sample_data()
    sp500_returns = handler.get_Ben_return()

    print("Données in-sample:\n", returns_in.head())
    print("\nDonnées out-of-sample:\n", returns_out.head())

    # Calcul des matrices de corrélation et de covariance
    cov_matrix = returns_in.cov()
    corr_matrix = returns_in.corr()

    hrp = HierarchicalRiskParity(corr_matrix)
    # Étape 1 : Conversion en matrice de distances condensée et clustering hiérarchique
    dist = hrp.correl_dist()
    dist_condensed = ssd.squareform(dist)
    link = sch.linkage(dist_condensed, 'single')

    # Étape 3 : Tri quasi-diagonal
    sortIx = hrp.get_quasi_diag(link)

    # Étape 4 : Allocation HRP
    weights_hrp = hrp.get_rec_bipart(cov_matrix, sortIx)

    markowitz = OptimizePortfolio(returns_in)
    lin_cost = np.full(returns_in.shape[1], 0.001)  # Coût linéaire
    risk_target = 0.02
    risk_aversion = 5

    weights_linear = markowitz.markowitz_linear(lin_cost, risk_target)
    weights_utility = markowitz.markowitz_utility(risk_aversion)
    weights_erc = markowitz.compute_ERC()
    weights_equal = np.ones(len(tickers)) / len(tickers)

    # Backtest out-of-sample
    returns_out_cum = pd.DataFrame({
        "Markowitz Transaction Cost": (returns_out @ weights_linear).cumsum(),
        "Markowitz Utility": (returns_out @ weights_utility).cumsum(),
        "ERC": (returns_out @ weights_erc).cumsum(),
        "Equal Weight": (returns_out @ weights_equal).cumsum(),
        "HRP Weight": (returns_out @ weights_hrp).cumsum()
    })

    # Backtest in-sample
    returns_in_cum = pd.DataFrame({
        "Markowitz Transaction Cost": (returns_in @ weights_linear).cumsum(),
        "Markowitz Utility": (returns_in @ weights_utility).cumsum(),
        "ERC": (returns_in @ weights_erc).cumsum(),
        "Equal Weight": (returns_in @ weights_equal).cumsum(),
        "HRP Weight": (returns_in @ weights_hrp).cumsum(),
        "S&P 500": sp500_returns.loc[returns_in.index].cumsum()
    })

    # Visualisation des performances In-Sample
    plt.figure(figsize=(20, 10))
    for col in returns_in_cum.columns:
        plt.plot(1 + returns_in_cum[col], label=col)
    plt.title("Performances In-Sample des stratégies")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulé")
    plt.legend()
    plt.grid()
    plt.show()

    # Visualisation des performances Out-of-Sample
    plt.figure(figsize=(20, 10))
    for col in returns_out_cum.columns:
        plt.plot(1 + returns_out_cum[col], label=col)
    plt.title("Performances Out-of-Sample of Asset Allocation without rebal")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulé")
    plt.legend()
    plt.grid()
    plt.savefig("performances_out_of_sample.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Appliquer la fonction aux rendements journaliers
    stats = pd.DataFrame({col: performance_stats(returns_out_cum[col].diff().dropna())
                          for col in returns_out_cum.columns}).T

    # Afficher les statistiques
    stats['Methods'] = stats.index
    print("Statistiques de performance :")
    print(stats)


    # Créer l'image du tableau
    fig, ax = plt.subplots(figsize=(6, 2))  # Ajustez la taille pour une meilleure visibilité
    ax.axis('off')  # Supprime les axes
    table = ax.table(
        cellText=stats.values,  # Contenu des cellules
        colLabels=stats.columns,  # Noms des colonnes
        loc='center',  # Emplacement du tableau
        cellLoc='center',  # Centrer le texte dans les cellules
    )

    # Ajuster la taille du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(stats.columns))))  # Ajuste les largeurs de colonnes

    # Sauvegarder le tableau en image
    plt.savefig("tableau.png", bbox_inches='tight', dpi=300)
    plt.show()



