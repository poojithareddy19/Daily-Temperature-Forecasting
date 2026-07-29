import pandas as pd
def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    n_test = max(1, int(len(df) * test_size))
    train = df.iloc[:-n_test].reset_index(drop=True)
    test = df.iloc[-n_test:].reset_index(drop=True)
    return train, test