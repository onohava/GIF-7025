import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class EarthQuakeDataLoader:
    @staticmethod
    def _feature_engineering(df, n_clusters=3):
        # 1. Create Timestamp feature (seconds)
        df['timestamp'] = df['dt_obj'].astype('int64') // 10 ** 9

        # 2. Create Location Cluster feature using K-Means (Latitude & Longitude)
        print(f"Clustering locations into {n_clusters} regions...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
        return df

    @staticmethod
    def load_and_prep_data(filepath, lookback=1, test_split=0.2, n_clusters=5):
        df = pd.read_csv(filepath)

        # 1. Standardize Column Names - as I am working with different datasets
        if 'mag' in df.columns:
            df = df.rename(columns={'mag': 'magnitude', 'type': 'data_type'})
        elif 'magnitudo' in df.columns:
            df = df.rename(columns={'magnitudo': 'magnitude'})

        # 2. Handle Time/Date
        if df['time'].dtype == 'object':
            df['dt_obj'] = pd.to_datetime(df['time'], format='mixed')
        elif 'date' in df.columns:
            df['dt_obj'] = pd.to_datetime(df['date'], format='mixed')

        # 3. Sort Chronologically
        df = df.sort_values(by='dt_obj', ascending=True)

        # 4. Filter Data
        if 'data_type' in df.columns:
            df = df[df['data_type'] == 'earthquake']
        if 'status' in df.columns:
            df = df[df['status'] == 'reviewed']

        df = df[df['magnitude'] > 5.0].copy()

        df = EarthQuakeDataLoader._feature_engineering(df, n_clusters)

        # Select features: Cluster, Depth, Magnitude, Timestamp
        feature_cols = ['location_cluster', 'depth', 'magnitude', 'timestamp']

        # Drop missing values
        df = df.dropna(subset=feature_cols)

        data = df[feature_cols].values

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
            # separate Scaler for TARGET (Magnitude is at index 2)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(data[:, 2].reshape(-1, 1))

        # Split into Train/Test
        train_size = int(len(data) * (1 - test_split))
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]

        def create_sequences(dataset, lookback):
            X, y = [], []
            for i in range(len(dataset) - lookback):
                X.append(dataset[i:i + lookback])
                # Target is Magnitude (Index 2)
                y.append(dataset[i + lookback, 2])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data, lookback)
        X_test, y_test = create_sequences(test_data, lookback)

        print(f"Data Loaded. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

        # Return target_scaler so you can visualize real values later
        return X_train, y_train, X_test, y_test, len(feature_cols), target_scaler