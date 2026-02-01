from source.config_manager import ConfigManager
import pandas as pd
import logging

class PremiumRepository:
    def __init__(self, config_manager: ConfigManager, dataframe: pd.DataFrame = None):
        self.config_manager = config_manager
        if dataframe is None:
            raise ValueError("PremiumRepository requires a dataframe input.")
        self._df_premium = dataframe.copy()
        self.update_date_format()

    @classmethod
    def from_dataframe(cls, config_manager: ConfigManager, dataframe: pd.DataFrame):
        if dataframe is None:
            raise ValueError("No dataframe provided to load premium data from.")

        df_premium = dataframe.copy()

        if not set([
            'UnderwritingYear',
            'Premium',
        ]).issubset(df_premium.columns):
            missing = set([
            'UnderwritingYear',
            'Premium',
        ]) - set(df_premium.columns)
            raise ValueError(f"Dataframe is missing required columns: {missing}")

        df_premium.rename(
            columns={
                'UnderwritingYear': 'uw_year',
                'Premium': 'Premium_selected',
            },
            inplace=True,
        )

        df_premium = df_premium[['uw_year', 'Premium_selected']].copy()
        # add  'GWP', 'EPI', 'GWP_Forecast' as seleced value columns for compatibility with Triangle
        for col in ['GWP', 'EPI', 'GWP_Forecast']:
            df_premium[col] = df_premium['Premium_selected'].copy()


        return cls(config_manager, df_premium)

    def get_premium(self):
        return self._df_premium.copy()

    def set_premium(self,df):
        self._df_premium = df.copy()

    def update_date_format(self):
        logging.info(self._df_premium)

        self._df_premium["uw_year"] = pd.to_datetime(
            self._df_premium["uw_year"].astype(str) + "-01-01"
        )
        self._df_premium["period"] = self._df_premium[
            "uw_year"
        ] + pd.offsets.QuarterEnd(
            0
        )  # set for gwp to first qtr (no developmenmt)

        logging.info(self._df_premium)
