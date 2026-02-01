from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

class ClaimsCollection:
    """
    Collection of all claims in long format  
    """
    
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._df['incurred'] = self._df['paid'] + self._df['outstanding']
        
    def iter_claims(self):
        """Make possible to iterate over claim df rather than object"""
        for id, df in self._df.groupby('id'):
            yield id, df

    def get_claim_amounts(self) -> pd.Series: 
        """Returns the total incurred amount for each claim."""
        return self._df.groupby('id')['incurred'].sum()
    
    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()
