import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Visualize():

    @staticmethod
    def plot_pval_matrix(pval_matrix, threshold=0.05):
        """
        Plots a heatmap for the p-values matrix from the cointegration test, highlighting significant values.

        Parameters:
            pval_matrix (pd.DataFrame): Matrix of p-values.
            threshold (float): Significance level to highlight values (default = 0.05).
        """
        plt.figure(figsize=(8, 6))

        # Create a mask for non-significant values
        mask = pval_matrix >= threshold  

        # Custom colormap: yellow for significant values, dark purple for others
        cmap = sns.color_palette(["yellow", "darkblue"])  

        # Plot heatmap with mask
        sns.heatmap(pval_matrix, annot=True, fmt=".3f", cmap="viridis", cbar=True,
                    linewidths=0.5, linecolor="gray", mask=mask)

        # Overlay a red border for significant values
        for i in range(pval_matrix.shape[0]):
            for j in range(pval_matrix.shape[1]):
                if pval_matrix.iloc[i, j] < threshold:
                    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f"Cointegration P-Values (Highlighted < {threshold})")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tickers(data, ticker1, ticker2, date_col="Date"):
        """
        Plots the prices of two specified tickers from a wide-format dataset with fewer x-ticks for better readability.
        """
        # Ensure the date column is set as the index
        if date_col in data.columns:
            data = data.set_index(date_col)

        # Check if the tickers exist in the dataset
        if ticker1 not in data.columns or ticker2 not in data.columns:
            raise ValueError(f"One or both tickers ({ticker1}, {ticker2}) not found in the dataset.")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data[ticker1], label=ticker1, linestyle="-", color="blue")
        ax.plot(data.index, data[ticker2], label=ticker2, linestyle="-", color="orange")

        # Reduce the number of x-axis labels
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))  # Adjust the number of ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as Year-Month

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add titles and labels
        ax.set_title(f"Price Comparison: {ticker1} vs {ticker2}", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Price", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spread(data, ticker1, ticker2, beta_matrix, date_col="Date"):
        """
        Plots the price spread of the form: ticker1 - beta * ticker2.
        """

        # Ensure the date column is set as the index
        if date_col in data.columns:
            data = data.set_index(date_col)
        
        # Check if the tickers exist in the dataset
        if ticker1 not in data.columns or ticker2 not in data.columns:
            raise ValueError(f"One or both tickers ({ticker1}, {ticker2}) not found in the dataset.")
        
        # Check if beta for the pair exists in the beta_matrix
        if ticker1 not in beta_matrix.columns or ticker2 not in beta_matrix.index:
            raise ValueError(f"Beta coefficient for pair ({ticker1}, {ticker2}) not found in the beta matrix.")
        
        # Get the beta coefficient for the pair
        beta = beta_matrix.loc[ticker2, ticker1]  # Assuming beta_matrix is in a pairwise format

        # Calculate the spread: ticker1 - beta * ticker2
        spread = data[ticker1] - beta * data[ticker2]

        # Compute the average spread
        avg_spread = spread.mean()

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, spread, label=f"Spread: {ticker1} - {beta:.2f} * {ticker2}", color="green")

        # Add a red dashed line for the average spread
        ax.axhline(avg_spread, color="red", linestyle="dashed", linewidth=2, label=f"Avg Spread: {avg_spread:.2f}")

        # Reduce the number of x-axis labels for better readability
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))  # Controls number of ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Formats as Year-Month

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add titles and labels
        ax.set_title(f"Price Spread: {ticker1} - {beta:.2f} * {ticker2}", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Spread", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()
