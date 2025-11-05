import train
import visualization
import time


def main_process():
    LOOKBACK_VALUES_TO_TEST = [10]

    print("=========================================")
    print("   STEP 1: STARTING MODEL EXPERIMENTS   ")
    print(f"  Will test {len(LOOKBACK_VALUES_TO_TEST)} lookback values: {LOOKBACK_VALUES_TO_TEST}")
    print("=========================================\n")

    for lb in LOOKBACK_VALUES_TO_TEST:
        print(f"\n--- STARTING RUN FOR LOOKBACK = {lb} ---")

        print(f"--- [PHASE 1 of 2] STARTING MODEL TRAINING ---")
        start_train = time.time()
        try:
            train.main(lookback_value=lb)
            end_train = time.time()
            print(f"--- [PHASE 1 of 2] TRAINING COMPLETE (Duration: {end_train - start_train:.2f}s) ---")
        except Exception as e:
            print(f"*** ERROR IN TRAINING PHASE (Lookback={lb}): {e} ***")
            continue

        print(f"\n--- [PHASE 2 of 2] STARTING MODEL VISUALIZATION ---")
        try:
            visualization.main(lookback_value=lb, all_lookbacks=LOOKBACK_VALUES_TO_TEST)
        except Exception as e:
            print(f"*** ERROR IN VISUALIZATION PHASE (Lookback={lb}): {e} ***")
            continue

    print("\n=========================================")
    print("           ALL EXPERIMENTS FINISHED           ")
    print("=========================================")

    print("\n=========================================")
    print("   STEP 2: CREATING FINAL SUMMARY PLOTS   ")
    print("=========================================")

    visualization.main(lookback_value=0, run_summary=True, all_lookbacks=LOOKBACK_VALUES_TO_TEST)

    print("\n=========================================")
    print("           PIPELINE FINISHED           ")
    print("=========================================")


if __name__ == "__main__":
    main_process()