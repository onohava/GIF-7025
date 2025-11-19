from data.data_loader import DataLoader
from learning.standard_earthquake_regressor import StandardEarthquakeRegressor
from constants import TARGET_FEATURE, TRAINING_FEATURES, DATA_FILEPATH, NUMBER_OF_EPOCHS, LEARNING_RATE
import matplotlib.pyplot as plt

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

    print("========== STARTING VISUALIZATION ==========")
    # Training loss
    plt.plot(range(len(model.training_losses)), model.training_losses, marker="o", linestyle="-")
    plt.title("Multi-layer Perceptron Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Prediction VS True values
    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions, alpha=0.5, color="blue", edgecolors="k")
    plt.title("Comparison of Predicted and True Magnitude Values")
    plt.xlabel("True Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.grid(True)
    plt.show()

    print("========== FINISHED VISUALIZATION ==========\n")

if __name__ == "__main__":
    run()