from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from random_forest_experiment.grid_optimisation import grid_optimisation

def main_process():
    print("=========================================")
    print("   STEP 1: STARTING MODEL EXPERIMENTS   ")
    print("=========================================\n")

    data = EarthQuakeDataLoader.load_and_prep_data(DATA_FILEPATH, 10, test_split=TEST_SPLIT)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)

    best_rf = grid_optimisation(rf, X_train, y_train)

    y_pred = best_rf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))



if __name__ == "__main__":
    main_process()