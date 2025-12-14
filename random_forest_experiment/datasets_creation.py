from typing import List, Dict, Callable, Any
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
min_magnitude = 0
max_magnitude = 10
min_longitude = -180
max_longitude = 180
min_latitude = -90
max_latitude = 90
min_depth = 0
shuffle_seed = 42

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
def create_filtered_dataset2(dataset, remove_features: List[str], rename_features: Dict[str, str], filter_dataset: Callable = filter_dataset):
    dataset.rename(columns=rename_features, inplace=True)
    dataset.drop(columns=remove_features, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset

def create_dataset(dataset, dataset_path: str, create_dataset: Callable, create_dataset_params: Dict[str, Any], load_dataset: bool = True, save_dataset: bool = True):
    print(f"Start of creation of dataset ({dataset_path})")
    if load_dataset: dataset = pd.read_csv(dataset)

    dataset = create_dataset(dataset, **create_dataset_params)

    if save_dataset: dataset.to_csv(dataset_path, index = False)

    print(f"End of creation of dataset ({dataset_path})")

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

def split_X_y(X, y, train_frac=0.6, test_frac=0.2, eval_frac=0.2, random_state=42):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_frac, random_state=random_state, shuffle=True
    )

    eval_frac_relative = eval_frac / (test_frac + eval_frac)

    X_test, X_eval, y_test, y_eval = train_test_split(
        X_temp, y_temp, test_size=eval_frac_relative, random_state=random_state, shuffle=True
    )

    return X_train, X_test, X_eval, y_train, y_test, y_eval
