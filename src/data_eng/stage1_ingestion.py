import sys
import pandas as pd
import argparse
import os

import hydra
from omegaconf import DictConfig

from app_logging import logging
from app_exception.exception import AppException
from data_eng.stage0_loading import GetData

class LoadData:
    '''
    The main functionality of this class is to load the data to the project folder path
    function return data and save it to folder we have assigned
    '''

    def __init__(self):
        self.getdata = GetData()

    def load_data(self, config): # input_path, output_path):
            
        try:
            logging.info(f"Loading data from the source")
            
            self.data = self.getdata.get_data(config)

            #print(config.data_source.external_data_dir + "/" + config.data_source.filename)

            self.raw_data_path = config.raw_data.raw_data_dir + "/" + config.raw_data.raw_filename
            self.data.to_csv(self.raw_data_path, sep=',', encoding='utf-8', index=False)
            logging.info(f"Data Loaded from the source successfully !!!")
        
        except Exception as e:
            logging.info(
                f"Exception Occurred while loading data from the source -->{e}")
            raise AppException(e, sys) from e


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    LoadData().load_data(cfg)
    ##data = GetData().get_data(cfg)
    #data_loader = GetData(cfg)
    #data = data_loader.get_data()

if __name__ == "__main__":
    main()



#if __name__ == '__main__':
#    args = argparse.ArgumentParser()
#    args.add_argument('--input_path', default='data/external/Consignment_pricing.csv', help='Ruta del archivo CSV de entrada usado para entrenar el modelo de predicción')
#    args.add_argument('--output_path', default='data/raw/dataset.csv', help='Ruta de salida para guardar los datos')

#    parsed_args = args.parse_args()
    
#    LoadData().load_data(input_path=parsed_args.input_path, output_path=parsed_args.output_path)


