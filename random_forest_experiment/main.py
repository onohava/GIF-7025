from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from data_reader import DataReader
from data_writer import DataWriter
from grid_optimisation import grid_optimisation

def main_process():
    print("=========================================")
    print("   STEP 1: STARTING MODEL EXPERIMENTS   ")
    print("=========================================\n")

    DataWriter.write()
    X_train, y_train, X_test, y_test, _ = DataReader.load_and_prep_data()

    best_params = {
    'bootstrap': False,
    'criterion': 'squared_error',
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 100
    }

    # Create Random Forest with these parameters
    rf = RandomForestRegressor(**best_params, random_state=42)

    # --- OPTIMIZE HYPERPARAMS ---
    # rf = grid_optimisation(rf, X_train, y_train)

    # --- TRAIN THE MODEL ---
    rf.fit(X_train, y_train)

    # --- MAKE PREDICTIONS ---
    y_pred = rf.predict(X_test)

    # --- EVALUATE ---
    print("R2:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    main_process()