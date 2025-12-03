from typing import List, Dict, Callable, Any
import pandas as pd

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
    "Datasets/Earthquakes-1990-2023.csv"
]

remove_features = {
    datasets_paths[0]: ["id", "url"],
    datasets_paths[1]: ["time", "state", "status", "tsunami", "significance", "data_type"]
}

rename_features = {
    datasets_paths[0]: {"mag": "magnitude", "depth_km": "depth", "time_utc": "date"},
    datasets_paths[1]: {"magnitudo": "magnitude"}
}

datasets_filtered_paths = {dataset_path: dataset_path.replace(".csv", "-filtered.csv") for dataset_path in datasets_paths}

datasets_filtered_subsets_sizes = {
    datasets_paths[0]: {"18K": 18000},
    datasets_paths[1]: {"1M": int(1e6), "2M": int(2e6), "3M": int(3e6)}
}

datasets_filtered_subsets_paths = {
    dataset_path: {
        subset_name: datasets_filtered_paths[dataset_path].replace(".csv", f"-{subset_name}.csv")
        for subset_name in datasets_filtered_subsets_sizes[dataset_path].keys()
    }
    for dataset_path in datasets_paths
}

# ---------- Methods ----------

def filter_dataset_feature(dataset, feature_name: str, min_value: float = float("-inf"), max_value: float = float("inf"), include_min_max: bool = True):
    if include_min_max:
        return dataset[(dataset[feature_name] >= min_value) & (dataset[feature_name] <= max_value)]
    else:
        return dataset[(dataset[feature_name] > min_value) & (dataset[feature_name] < max_value)]
    
def filter_dataset(dataset):
    dataset = filter_dataset_feature(dataset, "magnitude", min_magnitude, max_magnitude, False)
    dataset = filter_dataset_feature(dataset, "longitude", min_longitude, max_longitude, True)
    dataset = filter_dataset_feature(dataset, "latitude", min_latitude, max_latitude, True)
    dataset = filter_dataset_feature(dataset, "depth", min_depth, include_min_max=True)
    dataset["place"] = dataset["place"].map(lambda place: place.replace("CA", "California"))
    return dataset

def print_dataset(dataset_name: str, dataset):
    print(f"### {dataset_name} ###")
    print(dataset.info())
    print(dataset.describe())

def create_filtered_dataset(dataset, remove_features: List[str], rename_features: Dict[str, str], filter_dataset: Callable = filter_dataset):
    dataset.rename(columns=rename_features, inplace=True)
    dataset = filter_dataset(dataset)
    dataset.drop(columns=remove_features, inplace=True)
    dataset.drop_duplicates(inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset

def create_dataset(dataset, dataset_path: str, create_dataset: Callable, create_dataset_params: Dict[str, Any], load_dataset: bool = True, save_dataset: bool = True):
    print(f"Start of creation of dataset ({dataset_path})")
    # Load dataset
    if load_dataset: dataset = pd.read_csv(dataset)

    # Create dataset
    dataset = create_dataset(dataset, **create_dataset_params)

    # Save dataset
    if save_dataset: dataset.to_csv(dataset_path, index = False)

    print(f"End of creation of dataset ({dataset_path})")

    # Print dataset
    print_dataset(f"Dataset ({dataset_path.replace(".csv", "")})", dataset)

    return dataset


def create_subset(dataset, subset_size: int):
    subset = dataset.sample(n=min(subset_size, len(dataset)), random_state=shuffle_seed)
    subset.reset_index(drop=True, inplace=True)
    return subset

def filter_dataset2(dataset):
    dataset = dataset[dataset["data_type"] == "earthquake"]
    dataset = filter_dataset(dataset)
    return dataset

# Maps for dataset-specific filters
filter_datasets = {
    datasets_paths[0]: filter_dataset,
    datasets_paths[1]: filter_dataset2
}
