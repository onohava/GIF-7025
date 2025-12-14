from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from grid_optimisation import grid_optimisation
from datasets_creation import create_dataset, create_filtered_dataset, create_filtered_dataset2, filter_dataset, filter_dataset2, split_X_y
import pandas as pd
import joblib
import json

# Constants
min_magnitude = 0
max_magnitude = 10
min_longitude = -180
max_longitude = 180
min_latitude = -90
max_latitude = 90
min_depth = 0
shuffle_seed = 42

# Datasets paths
datasets_paths = [
    "Datasets/Earthquakes-180d.csv",
    "Datasets/Earthquakes-1990-2023.csv",
    "Datasets/X_test.csv"
]

remove_features = {
    datasets_paths[0]: ["id", "url"],
    datasets_paths[1]: ["time", "state", "status", "tsunami", "significance", "data_type"],
    datasets_paths[2]: ["time", "state", "status", "tsunami", "significance", "data_type"]
}

rename_features = {
    datasets_paths[0]: {"mag": "magnitude", "depth_km": "depth", "time_utc": "date"},
    datasets_paths[1]: {"magnitudo": "magnitude"},
    datasets_paths[2]: {"magnitudo": "magnitude"}
}

datasets_filtered_paths = {dataset_path: dataset_path.replace(".csv", "-filtered.csv") for dataset_path in datasets_paths}

datasets_filtered_subsets_sizes = {
    datasets_paths[0]: {"18K": 18000},
    datasets_paths[1]: {"1M": int(1e6), "2M": int(2e6), "3M": int(3e6)},
    datasets_paths[2]: {"183": 183}
}

datasets_filtered_subsets_paths = {
    dataset_path: {
        subset_name: datasets_filtered_paths[dataset_path].replace(".csv", f"-{subset_name}.csv")
        for subset_name in datasets_filtered_subsets_sizes[dataset_path].keys()
    }
    for dataset_path in datasets_paths
}

# Maps for dataset-specific filters
filter_datasets = {
    datasets_paths[0]: filter_dataset,
    datasets_paths[1]: filter_dataset2,
    datasets_paths[2]: filter_dataset2
}


def main_process():

    print("=========================================")
    print("     STEP 0: CREATE FILTERED DATASETS     ")
    print("=========================================\n")

    # Create filtered datasets BEFORE training anything
    for dataset_path in datasets_paths:
        output_path = datasets_filtered_paths[dataset_path]

        if dataset_path == "Datasets/X_test.csv":
            create_dataset(
            dataset=dataset_path,
            dataset_path=output_path,
            create_dataset=create_filtered_dataset2,
            create_dataset_params={
                "remove_features": remove_features[dataset_path],
                "rename_features": rename_features.get(dataset_path, {}),
                "filter_dataset": filter_datasets[dataset_path]

                }
            )
        else:

            create_dataset(
                dataset=dataset_path,
                dataset_path=output_path,
                create_dataset=create_filtered_dataset,
                create_dataset_params={
                    "remove_features": remove_features[dataset_path],
                    "rename_features": rename_features.get(dataset_path, {}),
                    "filter_dataset": filter_datasets[dataset_path]

                }
            )


    print("=========================================")
    print("   STEP 1: STARTING MODEL EXPERIMENTS   ")
    print("=========================================\n")

    df = pd.read_csv("Datasets/Earthquakes-1990-2023-filtered.csv")

    y = df["magnitude"]

    X = df.drop(columns=["magnitude"])
    X = X.drop(columns=["place", "date"])


    X_train, X_test, X_eval, y_train, y_test, y_eval = split_X_y(X, y)

    best_params = {
    'bootstrap': True,
    'criterion': 'squared_error',
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0,
    'min_samples_leaf': 2,
    'min_samples_split': 2
    }

    rf = RandomForestRegressor(**best_params, random_state=42, warm_start=True, n_jobs=-1,)

    subset_size = len(X_train) // 10

    for i in range(10):
        start = 0
        end = (i + 1) * subset_size if i < 9 else len(X_train)

        X_subset = X_train.iloc[start:end]
        y_subset = y_train.iloc[start:end]

        rf.n_estimators = 50 * (i + 1)

        rf.fit(X_subset, y_subset)

        y_pred_test = rf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"After subset {i+1}, estimator {rf.n_estimators}, trained on {end} samples, Test MSE: {mse_test:.4f}")

    y_pred_eval = rf.predict(X_eval)
    mse_eval = mean_squared_error(y_eval, y_pred_eval)
    r2_eval = r2_score(y_eval, y_pred_eval)

    print("Eval MSE:", mse_eval)
    print("Eval R²:", r2_eval)

    df = pd.read_csv("Datasets/X_test-filtered.csv")

    y = df["magnitude"]

    X = df.drop(columns=["magnitude"])
    X = X.drop(columns=["place", "date"])

    predictions = rf.predict(X)

    with open("predictions.json", "w") as f:
        json.dump(predictions.tolist(), f, indent=4)

    print(f"{len(predictions)} predictions written to predictions.json")


if __name__ == "__main__":
    main_process()