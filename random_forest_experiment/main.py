from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_reader import DataReader
from data_writer import DataWriter
from grid_optimisation import grid_optimisation

def main_process():
    print("=========================================")
    print("   STEP 1: STARTING MODEL EXPERIMENTS   ")
    print("=========================================\n")

    DataWriter.write()
    X_train, y_train, X_test, y_test, _ = DataReader.load_and_prep_data()

    rf = RandomForestClassifier()

    best_rf = grid_optimisation(rf, X_train, y_train)

    y_pred = best_rf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main_process()