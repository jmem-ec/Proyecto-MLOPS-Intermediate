import pandas as pd
import numpy as np
import argparse
import sys
import os

import hydra
from omegaconf import DictConfig

from app_logging import logging
from app_exception.exception import AppException

from data_eng.stage0_loading import GetData
from data_eng.stage1_ingestion import LoadData

class Preprocessing:
    '''
    This class is for preprocessing the data like column_imputation,missing_value_handling,transforming the
    columns and so on
    class return and save csv data to a specified folder path
    '''

    def __init__(self):
        self.load_data = LoadData()
        self.get_data = GetData()

    def column_imputation(self, config):
        try:
            logging.info("'column_imputation' FUNCTION STARTED")
            self.data = self.get_data.get_data(config)
            self.data.columns = self.data.columns.str.lower()
            self.data.columns = self.data.columns.str.replace(" ", "_")
            #self.data.columns = self.data.columns.str.replace("#", "")
            logging.info("'column_imputation' FUNCTION COMPILED SUCCESSFULLY")
            return self.data
        
        except Exception as e:
            logging.info(
                f"Exception occurred while compiling the code {str(e)}")
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

    def impute_missing(self, config):
        try:
            logging.info("'impute_missing' FUNCTION STARTED")
            self.data = self.column_imputation(config)
            self.data = self.data.drop("dosage", axis=1)
            self.data["shipment_mode"] = self.data["shipment_mode"].fillna("Air")
            self.data["line_item_insurance_(usd)"] = self.data["line_item_insurance_(usd)"].fillna(47.04)

            logging.info("'impute_missing' FUNCTION COMPILED SUCCESSFULLY")
            return self.data
        except Exception as e:
            logging.info(
                f"Exception occurred while compiling the code {str(e)}")
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

    def client_dates(self, date):
        if date == "Pre-PQ Process":
            return pd.to_datetime('01/06/2009', format="%d/%m/%Y")
        elif date == "Date Not Captured":
            return "Date Not Captured"
        else:
            if len(date) < 9:
                date = pd.to_datetime(date, format="%m/%d/%y")
                return date
            else:
                date = date.replace("-", "/")
                date = pd.to_datetime(date, format="%d/%m/%Y")
                return date

    def transform_pq_first_sent_to_client_date_columns(self, config):
        try:
            logging.info(
                "'transform_pq_first_sent_to_client_date_columns' FUNCTION STARTED")
            self.data = self.impute_missing(config)
            self.data["pq_first_sent_to_client_date"] = self.data["pq_first_sent_to_client_date"].apply(
                self.client_dates)
            self.data = self.data.drop(
                self.data.index[self.data["pq_first_sent_to_client_date"] == "Date Not Captured"])
            logging.info(
                "'transform_pq_first_sent_to_client_date_columns' FUNCTION COMPILED SUCCESSFULLY")
            return self.data
        except Exception as e:
            logging.info("Exception occurred while compiling the code", str(e))
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

    def transform_dates(self, data):
        data = data.replace("-", "/")
        data = pd.to_datetime(data, format="%d/%b/%y")
        return data

    def transform_dates_columns(self, config):
        try:
            logging.info("'transform_dates_columns' FUNCTION STARTED")
            self.data = self.transform_pq_first_sent_to_client_date_columns(
                config)
            self.data["delivery_recorded_date"] = self.data["delivery_recorded_date"].apply(
                self.transform_dates)
            self.data["delivered_to_client_date"] = self.data["delivered_to_client_date"].apply(
                self.transform_dates)
            self.data["pq_first_sent_to_client_date"] = pd.to_datetime(self.data["pq_first_sent_to_client_date"])
            
            self.data["days_to_process"] = self.data["delivery_recorded_date"]-self.data["pq_first_sent_to_client_date"]
            self.data["days_to_process"] = pd.to_timedelta(self.data["days_to_process"])
            self.data['days_to_process'] = self.data['days_to_process'].dt.days.astype(
                'int64')
            logging.info(
                "transform_dates_columns function compiled successfully")
            return self.data
        except Exception as e:
            logging.info("Exception occurred while compiling the code", str(e))
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e
        
    def reorder(self, data):
        data=data.split("-")
        data=data[0]
        return data   
     

    def trans_freight_cost(self, x):
        if x.find("See") != -1:
            return np.nan
        elif x == "Freight Included in Commodity Cost" or x == "Invoiced Separately":
            return 0
        else:
            return x

    def transform_freight_cost_columns(self, config):
        try:
            logging.info("'transform_freight_cost_columns' FUNCTION STARTED")
            self.data = self.transform_dates_columns(config)
            self.data = self.reorder_columns(config)
            self.data["freight_cost_(usd)"] = self.data["freight_cost_(usd)"].apply(
                self.trans_freight_cost)

            self.median_value = self.data["freight_cost_(usd)"].median()
            self.data["freight_cost_(usd)"] = self.data["freight_cost_(usd)"].replace(
                np.nan, self.median_value)
            logging.info(
                "transform_freight_cost_columns function compiled successfully")
            return self.data
        except Exception as e:
            raise AppException(e, sys) from e

    def drop_unnecessary_columns(self, config):
        try:
            logging.info("'drop_unnecessary_columns' FUNCTION STARTED")
            self.data = self.transform_dates_columns(config)
            delete_columns = ["pq_#",'po_/_so_#', 'asn/dn_#','country', 'fulfill_via', 'vendor_inco_term',
                'sub_classification', 'unit_of_measure_(per_pack)',
                'line_item_quantity', 'line_item_value', 'pack_price', 'unit_price',
                'first_line_designation', 'freight_cost_(usd)', 'shipment_mode',
                'line_item_insurance_(usd)', 'days_to_process']

            
            self.data = self.data[delete_columns]
            logging.info(
                "drop_unnecessary_columns function compiled successfully")
            return self.data
        except Exception as e:
            logging.info("Exception occurred while compiling the code", str(e))
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e
        

    def data_(self, config): #input_path, output_path):
        try:
            logging.info("'data' FUNCTION STARTED")
            #self.config = self.get_data.read_params(input_path)
            self.data = self.drop_unnecessary_columns(config)
            
            self.data["po_/_so_#"] = self.data["po_/_so_#"].apply(self.reorder)
            self.data["asn/dn_#"] = self.data["asn/dn_#"].apply(self.reorder)
            
            
            self.data.to_csv(config.cleaned_data.cleaned_data_dir + "/" + config.cleaned_data.cleaned_filename)
            logging.info("data function compiled successfully")
        except Exception as e:
            logging.info("Exception occurred while compiling the code")
            logging.info(
                "Failed to execute the code please check your code and run")
            raise AppException(e, sys) from e

@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    Preprocessing().data_(cfg) #input_path=args.input_path, output_path=args.output_path)

if __name__ == "__main__":
    main()



#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_path', default='data/raw/Consignment_pricing_raw.csv')
#    parser.add_argument('--input_path', default='data/raw/dataset.csv')
#    parser.add_argument('--output_path', default='data/interim/datacleaned.csv')
#    args = parser.parse_args()

#    Preprocessing().data_(input_path=args.input_path, output_path=args.output_path)
