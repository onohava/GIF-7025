import torch.nn as nn
from typing import List
from dataclasses import dataclass, field

@dataclass
class DataLoaderConfiguration():
    data_filepath: str = "standard_neural_network/data/Earthquakes-1990-2023.csv"
    target_feature: str = "magnitudo"
    training_features: List[str] = field(
        default_factory=lambda: ["longitude", "latitude", "depth"]
    )
    test_split: float = 0.2
    random_state: int = 42
    max_magnitude: float = 1000
    min_magnitude: float = 0


@dataclass
class TrainingConfiguration():
    loss_function: object = nn.MSELoss()
    random_state: int = 42
    number_of_epochs: int = 5
    learning_rate: float = 1E-3
    batch_size: int = 2048


@dataclass
class Configuration():
    data_loading: DataLoaderConfiguration 
    training: TrainingConfiguration


configurations: List[Configuration] = [
    Configuration(
        data_loading=DataLoaderConfiguration(),
        training=TrainingConfiguration()
    )
]
