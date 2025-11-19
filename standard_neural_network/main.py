from data.data_loader import DataLoader
from learning.standard_earthquake_regressor import StandardEarthquakeRegressor
from constants import TARGET_FEATURE, TRAINING_FEATURES, DATA_FILEPATH, NUMBER_OF_EPOCHS, LEARNING_RATE

def run():
    print("========== STARTING DATA EXTRACTION ==========")
    dataloader = DataLoader(target_feature=TARGET_FEATURE, training_features=TRAINING_FEATURES)
    dataloader.extract(data_filepath=DATA_FILEPATH).clean().transform()
    print("========== FINISHED DATA EXTRACTION ==========\n")

    X_train, X_test, y_train, y_test = dataloader.create_train_test_split(test_split=0.2)
    input_dimensions = X_train.shape[1]

    print("========== STARTING MODEL TRAINING ==========")
    model = StandardEarthquakeRegressor(input_dimensions=input_dimensions, learning_rate=LEARNING_RATE)
    model.train_model(X_train=X_train, y_train=y_train, epochs=NUMBER_OF_EPOCHS)
    print("========== FINISHED MODEL TRAINING ==========\n")

    print("========== STARTING MODEL TESTING ==========")
    model.evaluate(X_test=X_test, y_test=y_test)
    print("========== FINISHED MODEL TESTING ==========\n")

if __name__ == "__main__":
    run()