import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from standard_neural_network.configurations import DataLoaderConfiguration

class DataLoader:
    def __init__(self, data_loading_configuration: DataLoaderConfiguration):
        self.X = None
        self.y = None

        self.__config = data_loading_configuration
        self.__extracted_data = None
        self.__cleaned_data = None
        self.__transformer = None

    def extract(self):
        self.__extracted_data = pd.read_csv(self.__config.data_filepath) 
        print(f"Columns after extraction: {self.__extracted_data.columns.tolist()}")
        print(f"Rows after extraction: {len(self.__extracted_data)}")

        return self
    
    def clean(self):
        if self.__extracted_data is None:
            raise Exception("Data must be extracted with extract() before cleaning it.")
        
        self.__cleaned_data = self.__extracted_data.drop_duplicates()
        print(f"Rows after removing duplicates: {len(self.__cleaned_data)}")
        
        self.__cleaned_data = self.__cleaned_data[[self.__config.target_feature] + self.__config.training_features]

        self.__cleaned_data = self.__cleaned_data.dropna()
        print(f"Rows after removing missing values: {len(self.__cleaned_data)}")

        self.__cleaned_data = self.__cleaned_data[self.__cleaned_data[self.__config.target_feature] >= self.__config.min_magnitude]
        print(f"Rows after removing magnitudes below {self.__config.min_magnitude}: {len(self.__cleaned_data)}")

        self.__cleaned_data = self.__cleaned_data[self.__cleaned_data[self.__config.target_feature] <= self.__config.max_magnitude]
        print(f"Rows after removing magnitudes above {self.__config.max_magnitude}: {len(self.__cleaned_data)}")

        print(f"Columns after cleaning: {self.__cleaned_data.columns.tolist()}")

        return self
    
    def transform(self):
        if self.__cleaned_data is None:
            raise Exception("Data must be cleaned with clean() before transforming it.")
        
        self.X = self.__cleaned_data[self.__config.training_features]
        self.y = self.__cleaned_data[self.__config.target_feature]

        self.__transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.X.select_dtypes(include=["int64", "float64"]).columns.tolist()),
            ]
        )

        self.X = self.__transformer.fit_transform(self.X)
        self.y = self.y.to_numpy()

        return self

    def create_train_test_split(self):
        if self.X is None or self.y is None:
            raise Exception("Data must be transformed with transform() before creating test splits.")
        
        return train_test_split(self.X, self.y, test_size=self.__config.test_split, random_state=self.__config.random_state)
