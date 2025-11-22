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

    rf = RandomForestRegressor()

    best_rf = grid_optimisation(rf, X_train, y_train)

    y_pred = best_rf.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))


if __name__ == "__main__":
    main_process()