import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

from stage3_labeling import FeatureEngineering
from stage0_loading import GetData

class SplitData:
    """
    Simple version of SplitData for early-stage MLOps development. 
    - Splits into train and test
    - Saves to local CSV files
    """

    def __init__(self):
        self.get_data = GetData()
        self.labeling = FeatureEngineering()

    def split_data(self, input_path):
        print("Starting data splitting...")

        # Apply feature engineering (assumes it returns a transformed DataFrame)
        self.data = pd.read_csv(input_path, sep=",")
        print("Feature engineering retrieved.")

        # Split data
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)
        print("Data split into train and test sets.")

        # Save files locally (basic names for now)
        self.train.to_csv("data/processed/train.csv", sep=",",
                              index=False, encoding="UTF-8")
        self.test.to_csv("data/processed/test.csv", sep=",",
                              index=False, encoding="UTF-8")
        print("Train and test data saved to 'data/train.csv' and 'data/test.csv'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/processed/dataprocesed.csv')
    args = parser.parse_args()

    SplitData().split_data(input_path=args.input_path)

