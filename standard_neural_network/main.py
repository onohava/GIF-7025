from standard_neural_network.data.data_loader import DataLoader
from standard_neural_network.learning.standard_earthquake_regressor import StandardEarthquakeRegressor
import matplotlib.pyplot as plt
from standard_neural_network.configurations import Configuration, configurations
from dataclasses import asdict
import pprint

def run(configuration: Configuration):
    print("========== STARTING DATA EXTRACTION ==========")
    dataloader = DataLoader(configuration.data_loading)
    dataloader.extract().clean().transform()
    print("========== FINISHED DATA EXTRACTION ==========\n")

    X_train, X_test, y_train, y_test = dataloader.create_train_test_split()
    input_dimensions = X_train.shape[1]

    print("========== STARTING MODEL TRAINING ==========")
    model = StandardEarthquakeRegressor(training_configuration=configuration.training, input_dimensions=input_dimensions)
    model.train_model(X_train=X_train, y_train=y_train)
    print("========== FINISHED MODEL TRAINING ==========\n")

    print("========== STARTING MODEL TESTING ==========")
    model.evaluate(X_test=X_test, y_test=y_test)
    print("========== FINISHED MODEL TESTING ==========\n")

    print("========== STARTING VISUALIZATION ==========")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(range(len(model.training_losses)), model.training_losses, marker="o", linestyle="-")
    axes[0].set_title("Multi-layer Perceptron Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Prediction VS True values
    predictions = model.predict(X_test)
    axes[1].scatter(y_test, predictions, alpha=0.5, color="blue", edgecolors="k")
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_title("Comparison of Predicted and True Magnitude Values")
    axes[1].set_xlabel("True Magnitude")
    axes[1].set_ylabel("Predicted Magnitude")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print("========== FINISHED VISUALIZATION ==========\n")

if __name__ == "__main__":
    for configuration in configurations:
        print("========== RUNNING CONFIGURATION ==========")
        print(pprint.pp(asdict(configuration)), "\n")
        run(configuration)
        print("========== FINISHED RUNNING CONFIGURATION ==========")
