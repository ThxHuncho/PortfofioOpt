import matplotlib.pyplot as plt
import numpy as np

class Results:
    @staticmethod
    def display_weights(weights):
        print("Portfolio Weights:")
        for i, weight in enumerate(weights):
            print(f"Asset {i + 1}: {weight:.4f}")

    @staticmethod
    def plot_weights(weights, title="Portfolio Weights"):
        plt.figure(figsize=(20, 10))
        plt.bar(range(len(weights)), weights)
        plt.title(title)
        plt.xlabel("Assets")
        plt.ylabel("Weights")
        plt.show()

    @staticmethod
    def calculate_pnl(returns, weights):
        portfolio_returns = returns @ weights
        cumulative_pnl = np.cumsum(portfolio_returns)
        return cumulative_pnl

    @staticmethod
    def plot_pnl(dates, cumulative_pnl, title="Portfolio PnL"):
        """
        Plot portfolio cumulative PnL with dates on the x-axis.

        :param dates: list or np.ndarray
            Array of dates corresponding to returns.
        :param cumulative_pnl: np.ndarray
            Cumulative PnL data.
        :param title: str
            Plot title.
        """
        plt.figure(figsize=(20, 10))
        plt.plot(dates, cumulative_pnl)
        plt.title(title)
        plt.xlabel("Dates")
        plt.ylabel("Cumulative PnL")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    @staticmethod
    def display_results(dates_insample, returns_insample, 
                        dates_outsample, returns_outsample, weights):
        """
        Display portfolio weights and plot PnL in-sample and out-of-sample.

        :param dates_insample: list or np.ndarray
            Array of in-sample dates.
        :param returns_insample: np.ndarray
            In-sample returns matrix.
        :param dates_outsample: list or np.ndarray
            Array of out-of-sample dates.
        :param returns_outsample: np.ndarray
            Out-of-sample returns matrix.
        :param weights: np.ndarray
            Portfolio weights.
        """
        # Display weights
        Results.display_weights(weights)

        # Calculate and plot in-sample PnL
        pnl_insample = Results.calculate_pnl(returns_insample, weights)
        print("In-Sample PnL:")
        #print(pnl_insample)
        Results.plot_pnl(dates_insample, pnl_insample, title="In-Sample Portfolio PnL")

        # Calculate and plot out-of-sample PnL
        pnl_outsample = Results.calculate_pnl(returns_outsample, weights)
        print("Out-of-Sample PnL:")
        #print(pnl_outsample)
        Results.plot_pnl(dates_outsample, pnl_outsample, title="Out-of-Sample Portfolio PnL")
