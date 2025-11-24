import pandas as pd
import io
import os

from constants import DATA_FILEPATH, MOCK_CSV_DATA

class DataWriter:
    @staticmethod
    def write():
        if os.path.exists(DATA_FILEPATH):
            print(f"Mock data file '{DATA_FILEPATH}' already exists. Skipping creation.")
            return

        df = pd.read_csv(io.StringIO(MOCK_CSV_DATA))
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.to_parquet(DATA_FILEPATH)