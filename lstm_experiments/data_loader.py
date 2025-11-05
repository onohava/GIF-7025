import pandas as pd
import numpy as np



class EarthQuakeDataLoader:
    @staticmethod
    def create_sequences(data, lookback, target_col_index):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i, target_col_index])

        return np.array(X), np.array(y)

    @staticmethod
    def load_and_prep_data(filepath, lookback, target_col='magnitude', test_split=0.2):
        df = pd.read_parquet(filepath)

        target_col_index = df.columns.get_loc(target_col)

        data_values = df.values
        num_features = data_values.shape[1]

        X, y = EarthQuakeDataLoader.create_sequences(data_values, lookback, target_col_index)

        # split into train and test sets
        split_index = int(len(X) * (1 - test_split))

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Data shapes (Samples, Lookback, Features):")
        print(f"X_train: {X_train.shape}")
        print(f"X_test:  {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test:  {y_test.shape}")

        return X_train, y_train, X_test, y_test, num_features