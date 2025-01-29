import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint, adfuller
from scipy.spatial.distance import pdist, squareform

from itertools import combinations

import logging
from typing import List, Optional, Tuple
from pandas import DataFrame

from scripts.time_decorator import TimingMeta
    
    
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PairSelector(metaclass=TimingMeta):
    
    @classmethod
    def preprocess_data(cls, data: DataFrame, standardize: bool = False, log_version: bool = False) -> DataFrame:
        """Preprocess data by applying log transformation and/or standardization."""
        if log_version:
            logger.info("Applying log transformation to data.")
            data = np.log(data / data.shift(1)).dropna()

        if standardize:
            logger.info("Standardizing data.")
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

        return data
    
    def compute_pairwise_distance(self, data: DataFrame, standardize: bool = False, log_version: bool = False) -> DataFrame:
        """Compute the pairwise Euclidean distance between assets."""
        logger.info("Computing Euclidean distance matrix.")
        data = self.preprocess_data(data, standardize, log_version)
        
        # Compute pairwise distances using vectorized computation
        dist_array = pdist(data.T, metric='euclidean')
        distance_matrix = pd.DataFrame(squareform(dist_array), index=data.columns, columns=data.columns)

        return distance_matrix

    def compute_distance_correlation(self, data:DataFrame, standardize:Optional[bool]=False, log_version: Optional[bool]=False)->DataFrame:
        logger.info("Distance correlation")
        data = self.preprocess_data(data, standardize, log_version)

        
        correlation_matrix = data.corr()
        correlation_distance = 1 - correlation_matrix

        return correlation_distance
    
    def compute_cointegration_metrics(self, data:DataFrame, standardize:Optional[bool]=False, log_version: Optional[bool]=False)->Tuple[DataFrame,DataFrame]:
        logger.info("Integral correlation")
        data = self.preprocess_data(data, standardize, log_version)
        assets = data.columns
        n = len(assets)   
        
        beta_matrix = pd.DataFrame(np.nan, index=assets, columns=assets)
        p_value_matrix = pd.DataFrame(np.nan, index=assets, columns=assets)
        # Iterate only over the lower triangle (excluding diagonal)
        for i in range(n):
            for j in range(i + 1, n):  # j > i ensures each pair is computed only once
                asset_x, asset_y = assets[i], assets[j]

                # Step 1: Regression asset_y ~ beta * asset_x
                X = sm.add_constant(data[asset_x])  # Independent variable with intercept
                y = data[asset_y]  # Dependent variable
                model = sm.OLS(y, X).fit()
                beta = model.params[asset_x]

                # Store beta symmetrically
                beta_matrix.loc[asset_y, asset_x] = beta
                beta_matrix.loc[asset_x, asset_y] = 1 / beta if beta != 0 else np.nan  # Invert beta for symmetry

                # Step 2: Compute spread (residuals)
                spread = y - beta * data[asset_x]

                # Step 3: ADF test on spread
                adf_result = adfuller(spread.dropna())
                p_value = adf_result[1]  # Extract p-value

                # Store p-value symmetrically
                p_value_matrix.loc[asset_y, asset_x] = p_value
                p_value_matrix.loc[asset_x, asset_y] = p_value  # p-value is the same in both directions

        return p_value_matrix, beta_matrix
    
    @staticmethod
    def get_smallest_values(df: DataFrame, n:Optional[int]=10):
        """
        Extracts the smallest n values from the upper triangular part of a symmetric DataFrame,
        avoiding duplicate (i, j) and (j, i) pairs.

        Parameters:
            df (pd.DataFrame): Input symmetric DataFrame.
            n (int): Number of smallest values to extract.

        Returns:
            pd.DataFrame: A DataFrame containing the smallest n values and their row/column indices.
        """
        # Get the upper triangular part of the matrix, excluding the diagonal
        upper_tri = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
        
        # Flatten the DataFrame and drop NaN values (which come from lower triangle)
        flattened = upper_tri.stack()
        
        # Sort values and get the smallest n
        smallest = flattened.nsmallest(n)
        
        # Create a DataFrame with value, row, and column information
        result = pd.DataFrame({
            "value": smallest.values,
            "asset_1": [index[0] for index in smallest.index],
            "asset_2": [index[1] for index in smallest.index],
        })
        
        return result

    @staticmethod
    def get_values_below_threshold(df: DataFrame, threshold:Optional[float]=0.05):
        """
        Extracts all values below a given threshold from a DataFrame, along with their row and column indices.

        Parameters:
            df (pd.DataFrame): Input DataFrame representing the matrix.
            threshold (float): The threshold value to filter the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing values below the threshold and their row/column indices.
        """
        # Apply the mask for values below the threshold
        mask = (df < threshold) & np.tril(np.ones(df.shape), k=-1).astype(bool)
        filtered = df[mask]

        # Flatten the result and get row-column indices
        result = pd.DataFrame({
            "value": filtered.stack().values,
            "asset_1": [index[0] for index in filtered.stack().index],
            "asset_2": [index[1] for index in filtered.stack().index],
        })

        return result