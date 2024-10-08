def filt_ds(train_df, val_df, test_df, only_stage, only_brain_region):
    # train_df: pandas Dataframe with Training Set
    # val_df: pandas Dataframe with Validation Set
    # test_df: pandas Dataframe with Test Set
    # only_stage: sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM
    # only_brain_region: brain region to be considered for the analysis i.e., Fp, F, C, T, P, O

    if only_stage and only_brain_region:
        train_filt = train_df[(train_df.iloc[:, 2] == only_brain_region) &
                              (train_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
        val_filt = val_df[(val_df.iloc[:, 2] == only_brain_region) &
                          (val_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
        test_filt = test_df[(test_df.iloc[:, 2] == only_brain_region) &
                            (test_df.iloc[:, 3] == only_stage)].reset_index(drop=True)
    elif only_stage and not only_brain_region:
        train_filt = train_df[train_df.iloc[:, 3] == only_stage].reset_index(drop=True)
        val_filt = val_df[val_df.iloc[:, 3] == only_stage].reset_index(drop=True)
        test_filt = test_df[test_df.iloc[:, 3] == only_stage].reset_index(drop=True)
    elif not only_stage and only_brain_region:
        train_filt = train_df[train_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
        val_filt = val_df[val_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
        test_filt = test_df[test_df.iloc[:, 2] == only_brain_region].reset_index(drop=True)
    else:
        train_filt = train_df
        val_filt = val_df
        test_filt = test_df

    return train_filt, val_filt, test_filt