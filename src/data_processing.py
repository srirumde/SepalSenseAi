import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.processed_data_path = "artifacts/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error in loading data: {str(e)}")
            raise CustomException(f"Failed to read data: {e}")

    def handle_outliers(self, column):
        try:
            logger.info(f"Starting outlier handling for column: {column}")
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            median_value = self.df[column].median()

            # Replace outliers with median
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            self.df.loc[outliers, column] = median_value
            logger.info("Outliers handled successfully.")

        except Exception as e:
            logger.error(f"Error handling outliers in column {column}: {e}")
            raise CustomException(f"Failed to handle outliers in column '{column}': {e}")

    def split_data(self):
        try:
            X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            Y = self.df["Species"]

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            logger.info("Data split into training and test sets successfully.")

            # Save to artifacts
            joblib.dump(X_train, os.path.join(self.processed_data_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_data_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_data_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_data_path, "y_test.pkl"))
            logger.info("Train/test sets saved successfully.")

        except Exception as e:
            logger.error(f"Error in splitting data: {e}")
            raise CustomException(f"Failed to split data: {e}")

    def run(self):
        self.load_data()
        self.handle_outliers("SepalWidthCm")  # Corrected column name
        self.split_data()


if __name__ == "__main__":
    data_processor = DataProcessing("artifacts/raw/data.csv")
    data_processor.run()
