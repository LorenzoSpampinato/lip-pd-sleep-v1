import pandas as pd


class DataFrameFilter:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Initializes the DataFrameFilter with training, validation, and test DataFrames.

        Parameters:
        - train_df: pandas DataFrame with Training Set
        - val_df: pandas DataFrame with Validation Set
        - test_df: pandas DataFrame with Test Set
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def filter_dfs(self, only_stage: str = None, only_brain_region: str = None):
        """
        Filters the DataFrames based on specified sleep stage and brain region.

        Parameters:
        - only_stage: sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM
        - only_brain_region: brain region to be considered for the analysis i.e., Fp, F, C, T, P, O

        Returns:
        - A tuple of filtered DataFrames (train_filt, val_filt, test_filt).
        """
        # Apply filtering based on provided criteria
        if only_stage and only_brain_region:
            train_filt = self.train_df[(self.train_df.iloc[:, 2] == only_brain_region) &
                                       (self.train_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
            val_filt = self.val_df[(self.val_df.iloc[:, 2] == only_brain_region) &
                                   (self.val_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
            test_filt = self.test_df[(self.test_df.iloc[:, 2] == only_brain_region) &
                                     (self.test_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
        elif only_stage:
            train_filt = self.train_df[self.train_df.iloc[:, 3] == only_stage].reset_index(drop=True)
            val_filt = self.val_df[self.val_df.iloc[:, 3] == only_stage].reset_index(drop=True)
            test_filt = self.test_df[self.test_df.iloc[:, 3] == only_stage].reset_index(drop=True)
        elif only_brain_region:
            train_filt = self.train_df[self.train_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
            val_filt = self.val_df[self.val_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
            test_filt = self.test_df[self.test_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
        else:
            # No filtering applied
            train_filt = self.train_df
            val_filt = self.val_df
            test_filt = self.test_df

        return train_filt, val_filt, test_filt

# Example usage
# train_df = pd.DataFrame(...)  # your training DataFrame
# val_df = pd.DataFrame(...)    # your validation DataFrame
# test_df = pd.DataFrame(...)    # your test DataFrame

# df_filter = DataFrameFilter(train_df, val_df, test_df)
# filtered_train, filtered_val, filtered_test = df_filter.filter_dfs(only_stage='N2', only_brain_region='C')
