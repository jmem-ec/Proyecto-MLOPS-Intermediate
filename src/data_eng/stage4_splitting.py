import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import os

import hydra
from omegaconf import DictConfig

from data_eng.stage3_labeling import FeatureEngineering
from data_eng.stage0_loading import GetData

from app_logging import logging


class SplitData:
    """
    Simple version of SplitData for early-stage MLOps development. 
    - Splits into train and test
    - Saves to local CSV files
    """

    def __init__(self):
        self.get_data = GetData()
        self.labeling = FeatureEngineering()

    def split_data(self, config):
        logging.info("Starting data splitting...")

        # Apply feature engineering (assumes it returns a transformed DataFrame)
        self.data = pd.read_csv(config.processed_data.processed_data_dir + "/" + config.processed_data.processed_filename, sep=",")
        logging.info("Feature engineering retrieved.")

        # Split data
        self.train, self.test = train_test_split(self.data, test_size=config.splited_data.split_ratio, random_state=config.splited_data.random_state)
        logging.info("Data split into train and test sets.")

        # Save files locally (basic names for now)
        self.train.to_csv(config.splited_data.train_data_dir + "/" + config.splited_data.train_filename, sep=",",
                              index=False, encoding="UTF-8")
        self.test.to_csv(config.splited_data.test_data_dir + "/" + config.splited_data.test_filename, sep=",",
                              index=False, encoding="UTF-8")
        logging.info("Train and test data saved to 'data/train.csv' and 'data/test.csv'.")

@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    SplitData().split_data(cfg) 

if __name__ == "__main__":
    main()