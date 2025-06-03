import joblib
import pandas as pd
import sys
import os

import hydra
from omegaconf import DictConfig

from app_logging import logging
from app_exception.exception import AppException

class Predict:

    def __init__(self):
        pass

    def get_latest_model_path(self, model_dir, filename):
        try:
            folders = sorted(os.listdir(model_dir), reverse=True)
            for folder in folders:
                full_path = os.path.join(model_dir, folder, filename)
                if os.path.isfile(full_path):
                    return full_path            
        except Exception as e:
            logging.info("Exception occurred while get latest model" + str(e))
            logging.info("No trained model found.")
            raise AppException(e, sys) from e

    def predict(self, cfg):

        model_dir = cfg.model_data.models_dir
        filename = cfg.model_data.file_model

        model_path = self.get_latest_model_path(model_dir, filename)
        model = joblib.load(model_path)

        # Datos de prueba como diccionario
        sample_data = {
            "po_/_so_#": 0.00,
            "asn/dn_#": 0.00,
            "country": 38.00,
            "fulfill_via": 0.00,
            "vendor_inco_term": 5.00,
            "sub_classification": 5.00,
            "unit_of_measure_(per_pack)": 240.00,
            "line_item_quantity": 1000.00,
            "pack_price": 6.2,
            "unit_price": 0.03,
            "first_line_designation": 1.00,
            "freight_cost_(usd)": 4521.5,
            "shipment_mode": 0.00,
            "line_item_insurance_(usd)": 47.04,
            "days_to_process": -930
        }

        # Crear DataFrame de prueba
        sample_df = pd.DataFrame([sample_data])

        output = model.predict(sample_df)

        logging.info("Predicition successfully")
        logging.info(output)

@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="model_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    Predict().predict(cfg)

if __name__ == "__main__":
    main()
