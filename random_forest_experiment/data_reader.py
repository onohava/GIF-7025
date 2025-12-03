import numpy as np
import pandas as pd

from datasets_creation import (
    datasets_paths,
    datasets_filtered_paths,
    remove_features,
    rename_features,
    filter_datasets,
    create_dataset,
    create_filtered_dataset
)


class DataReader:
    @staticmethod
    def load_and_prep_data(dataset_index=0, target_col='magnitude', test_split=0.2):
        """
        Loads and filters the dataset from earthquake_data pipeline, then splits it.
        
        :param dataset_index: 0 for first dataset, 1 for second
        :param target_col: Column to predict
        :param test_split: Fraction of data to use as test set
        """
        dataset_path = datasets_paths[dataset_index]
        filtered_path = datasets_filtered_paths[dataset_path]

        # Apply your filtering / renaming / feature removal pipeline
        df = create_dataset(
            dataset_path,
            filtered_path,
            create_filtered_dataset,
            {
                "remove_features": remove_features[dataset_path],
                "rename_features": rename_features[dataset_path],
                "filter_dataset": filter_datasets[dataset_path]
            }
        )

        # Separate target and features
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values

        num_features = X.shape[1]

        # Train/test split
        split_index = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"Data shapes (Samples, Features):")
        print(f"X_train: {X_train.shape}")
        print(f"X_test:  {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test:  {y_test.shape}")

        return X_train, y_train, X_test, y_test, num_features

    @staticmethod
    def create_sequences(data, lookback, target_col_index):
        """
        Optional: Convert time series into sequences if needed.
        """
        X_seq, y_seq = [], []
        for i in range(lookback, len(data)):
            X_seq.append(data[i - lookback:i])
            y_seq.append(data[i, target_col_index])
        return np.array(X_seq), np.array(y_seq)