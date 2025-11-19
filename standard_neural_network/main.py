from data.data_loader import DataLoader
from constants import TARGET_FEATURE, TRAINING_FEATURES, DATA_FILEPATH

def run():
    dataloader = DataLoader(target_feature=TARGET_FEATURE, training_features=TRAINING_FEATURES)
    dataloader.extract(DATA_FILEPATH).clean().transform()

if __name__ == "__main__":
    run()