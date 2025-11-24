import pandas as pd
import numpy as np

from constants import DATA_FILEPATH



class DataReader:
    @staticmethod
    def create_sequences(data, lookback, target_col_index):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i, target_col_index])

        return np.array(X), np.array(y)

    @staticmethod
    def load_and_prep_data(target_col='magnitude', test_split=0.2):
        df = pd.read_parquet(DATA_FILEPATH)

        y = df[target_col].values
        X = df.drop(columns=[target_col]).values

        num_features = X.shape[1]

        split_index = int(len(X) * (1 - test_split))

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Data shapes (Samples, Features):")
        print(f"X_train: {X_train.shape}")
        print(f"X_test:  {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test:  {y_test.shape}")

        return X_train, y_train, X_test, y_test, num_features
