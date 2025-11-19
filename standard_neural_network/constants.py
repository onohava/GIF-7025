from data.data_loader import DataLoader

DATA_FILEPATH = "data/Earthquakes-1990-2023.csv"
TARGET_FEATURE = "magnitudo"
TRAINING_FEATURES = ["tsunami", "significance", "state", "longitude", "latitude", "depth", "date"]

dataloader = DataLoader(target_feature=TARGET_FEATURE, training_features=TRAINING_FEATURES)

dataloader.extract(DATA_FILEPATH).clean().transform()