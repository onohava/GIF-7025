import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class EarthQuakeDataLoader:
    @staticmethod
    def _feature_engineering(df, n_clusters=5):
        # 1. Create Timestamp feature (seconds)
        if 'date' in df.columns:
            df['dt_obj'] = pd.to_datetime(df['date'], format='mixed')
            df['timestamp'] = df['dt_obj'].astype('int64') // 10 ** 9

        # 2. Create Location Cluster feature using K-Means (Latitude & Longitude)
        print(f"Clustering locations into {n_clusters} regions...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

        return df

    @staticmethod
    def load_and_prep_data(filepath, lookback=1, test_split=0.2, n_clusters=5):
        df = pd.read_csv(filepath)
        df = df[df['magnitudo'] > 5.0].copy()

        # Feature Engineering
        df = EarthQuakeDataLoader._feature_engineering(df, n_clusters)

        # Select features specified in the article:
        # "Location cluster, depth, magnitude, and timestamp"
        feature_cols = ['location_cluster', 'depth', 'magnitudo', 'timestamp']
        data = df[feature_cols].values

        # Scaling (MinMax 0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Split into Train/Test
        train_size = int(len(data) * (1 - test_split))
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]

        # Create Sequences (t-1 predicts t)
        # Target is magnitude (index 2 in our feature_cols list)
        target_idx = 2

        def create_sequences(dataset, lookback):
            X, y = [], []
            for i in range(len(dataset) - lookback):
                X.append(dataset[i:i + lookback])
                y.append(dataset[i + lookback, target_idx])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data, lookback)
        X_test, y_test = create_sequences(test_data, lookback)

        print(f"Data Loaded. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
        return X_train, y_train, X_test, y_test, len(feature_cols), scaler