import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  # You'll need this import


# Assume this class is in 'data_loader.py'
class EarthQuakeDataLoader:

    @staticmethod
    def _prepare_data(filepath):
        df = pd.read_csv(filepath)
        df.drop_duplicates(keep='first', inplace=True)
        df = df[df['magnitudo'] > 4]

        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.set_index('date').sort_index()
        df = df.drop(columns=['time', 'place', 'status', 'data_type'])
        df = df.rename(columns={'magnitudo': 'magnitude'})
        df = pd.get_dummies(df, columns=['state'])

        print(f"Dataset Length: {len(df)}")
        return df

    @staticmethod
    def _create_sequences(df, all_features, target_col, lookback):
        X, y = [], []

        feature_data = df[all_features].values
        target_data = df[target_col].values

        for i in range(len(feature_data) - lookback):
            X.append(feature_data[i:(i + lookback)])
            y.append(target_data[i + lookback])

        return np.array(X), np.array(y)

    @staticmethod
    def load_and_prep_data(filepath, lookback, test_split=0.2):
        df = EarthQuakeDataLoader._prepare_data(filepath)

        all_features = list(df.columns)
        target_col = 'magnitude'
        num_features = len(all_features)

        train_df, test_df = train_test_split(df, test_size=test_split, shuffle=False)
        scaler = MinMaxScaler()

        train_df_scaled = train_df.copy()
        test_df_scaled = test_df.copy()

        train_df_scaled[all_features] = scaler.fit_transform(train_df[all_features])

        test_df_scaled[all_features] = scaler.transform(test_df[all_features])

        X_train, y_train = EarthQuakeDataLoader._create_sequences(
            train_df_scaled, all_features, target_col, lookback
        )
        X_test, y_test = EarthQuakeDataLoader._create_sequences(
            test_df_scaled, all_features, target_col, lookback
        )

        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

        return X_train, y_train, X_test, y_test, num_features