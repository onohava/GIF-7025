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
    run()