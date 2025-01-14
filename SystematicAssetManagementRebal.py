from SystematicAM import *

if __name__ == "__main__":
    # Configuration des tickers et des dates
    #tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL","GOOG", "WMT", "UNH"]
    tickers = ["XOM", "MSFT", "C", "GE", "WMT", "BAC", "JNJ", "PFE", "INTC", "AIG"]

    benchmark_ticker = '^GSPC'
    start_date = "2000-01-01"
    end_date = "2025-01-01"


    # Création de l'instance pour manipuler les données
    handler = FinancialDataHandler(tickers, benchmark_ticker, start_date, end_date, None)
    handler.process()

    # Récupérer les rendements
    returns = handler.returns
    sp500_returns = handler.sp500_returns

    # Définition des méthodes d'optimisation
    optimization_methods = {
        "Markowitz Linear Cost": lambda optimizer: optimizer.markowitz_linear(
            lin_cost=np.full(returns.shape[1], 0.001), risk_target=0.02
        ),
        "Markowitz Utility": lambda optimizer: optimizer.markowitz_utility(risk_aversion=5),
        "Equal Risk Contribution (ERC)": lambda optimizer: pd.Series(
            optimizer.compute_ERC(), index=returns.columns
        ),
    }

    # Ajouter HRP comme méthode distincte
    def hrp_method(returns):
        corr_matrix = returns.corr()
        cov_matrix = returns.cov()
        hrp = HierarchicalRiskParity(corr_matrix)
        dist = hrp.correl_dist()
        dist_condensed = ssd.squareform(dist)
        link = sch.linkage(dist_condensed, 'single')
        sort_ix = hrp.get_quasi_diag(link)
        weights_hrp = hrp.get_rec_bipart(cov_matrix, sort_ix)
        return weights_hrp

    optimization_methods["Hierarchical Risk Parity (HRP)"] = lambda optimizer: hrp_method(optimizer.returns)

    # Exécution des backtests pour chaque méthode
    start_year = 2005
    window_size = 10  # 3 ans in-sample
    step_size = 1  # 1 an out-of-sample

    # Dictionnaire pour stocker les performances cumulées
    all_performances = {}

    for method_name, method in optimization_methods.items():
        print(f"Backtest pour la méthode : {method_name}")
        backtester = Backtester(returns, method)
        weights_df, combined_performance = backtester.run_backtest(start_year, window_size, step_size)

        # Stocker les performances
        all_performances[method_name] = combined_performance

        # Afficher quelques résultats
        print("Poids calculés pour chaque fenêtre:", weights_df.head())
        print("Performance cumulée:", combined_performance.head())


    returns_out_cum = pd.DataFrame({method_name: combined_performance for method_name, combined_performance in all_performances.items()})
    print(returns_out_cum)

    # Tracer les performances cumulées pour toutes les méthodes
    plt.figure(figsize=(20, 10))
    for method_name, combined_performance in all_performances.items():
        plt.plot(1+combined_performance, label=method_name)
    plt.title("Performances cumulées des différentes stratégies")
    plt.xlabel("Date")
    plt.ylabel("Performance cumulée")
    plt.legend()
    plt.grid(False)
    plt.savefig("performances_out_of_sampleRebal.png", dpi=300, bbox_inches='tight')
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
        cellLoc='center'  # Centrer le texte dans les cellules
    )

    # Ajuster la taille du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(stats.columns))))  # Ajuste les largeurs de colonnes

    # Sauvegarder le tableau en image
    plt.savefig("tableauRebal.png", bbox_inches='tight', dpi=300)
    plt.show()
