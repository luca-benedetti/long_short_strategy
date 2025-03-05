# input: spread or price data (need only the pairs to trade), pairs indication (maybe beta too)
# output: set of signals -1 0 1
# 1 simple: take spread, evaluate mean, variance and generate signal based on deviations
# 2 medium: implement spread fitting and generate optimal entering/exit strategy
# functions:
# - re evaluate beta, check

from typing import List, Optional
from pydantic import BaseModel

import pandas as pd
from pandas import DataFrame
import numpy as np
# Date | Asset1 | Asset2 | Spread | Trading Signal


class AssetPair(BaseModel):
    asset1: str
    asset2: str
    beta: float

class Strategy():
    
    def evaluate_spreads(self, df:DataFrame, pairs:List[AssetPair],k:Optional[float]=1.5):
        pairs_dfs = []
        for pair in pairs:
            df_pair = df[[f"{pair.asset1}",f"{pair.asset2}"]]
            df_pair = self.evaluate_spread(df_pair,pair)
            df_pair = self.generate_signal(df_pair,k=k)
            df_pair = self.shift_signal(df_pair)
            pairs_dfs.append(df_pair)
        return pairs_dfs

    def evaluate_spread(self, df:DataFrame, pair: AssetPair):
        df["Spread"] = df[f"{pair.asset1}"]-pair.beta*df[f"{pair.asset2}"]
        df["Spread"] -= df.Spread.mean()
        return df
    
    def generate_signal(self, df: pd.DataFrame, k: Optional[float] = 1.5) -> pd.DataFrame:
        # TODO: allow also more flexible output position and eventually stop losses
        mean = df.Spread.mean()
        std = df.Spread.std()
        upper_threshold = mean + k * std
        lower_threshold = mean - k * std
        
        signals = np.zeros(len(df))
        position = 0  # 1 for long, -1 for short, 0 for no position
        
        for i in range(len(df)):
            if position == 0:
                if df.Spread.iloc[i] > upper_threshold:
                    signals[i] = -1  # Enter short
                    position = -1
                elif df.Spread.iloc[i] < lower_threshold:
                    signals[i] = 1  # Enter long
                    position = 1
            elif position == 1 and df.Spread.iloc[i] >= 0:
                signals[i] = 0  # Exit long
                position = 0
            elif position == -1 and df.Spread.iloc[i] <= 0:
                signals[i] = 0  # Exit short
                position = 0
            else:
                signals[i] = position  # Maintain position
        
        df["Signal"] = signals
        return df
    
    def shift_signal(self, df:DataFrame):
        df["Signal_Shifted"] = df["Signal"].shift(1)
        return df
        