import os
import json
import joblib
import sys
import pandas as pd
import numpy as np
import os

import hydra
from omegaconf import DictConfig

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from app_logging import logging
from app_exception.exception import AppException

from data_eng.stage0_loading import GetData


class TrainEvaluate:
    def __init__(self, config):
        self.get_data = GetData()
        self.filename = config.model_data.file_model

    def evaluation_metrics(self, act, pred):
        self.r2_score = r2_score(act, pred)
        self.mse = mean_squared_error(act, pred)
        self.rmse = np.sqrt(mean_squared_error(act, pred))
        return self.r2_score, self.mse, self.rmse

    def model_eval(self, config):
        try:
            logging.info("'train_evaluate' function started")
            self.test_data = config.model_data.train_data_dir + "/" + config.model_data.train_filename #"data/processed/test.csv"
            self.train_data = config.model_data.test_data_dir + "/" + config.model_data.test_filename #"data/processed/train.csv"

            # Recupera directorio para almacenar el modelo entrenado
            self.model_dir = config.model_data.models_dir
            MODEL_PATH = os.path.join(os.getcwd(), self.model_dir)

            # Recupera directorio para almacenar los reportes
            self.report_dir = config.model_data.reports_dir
            REPORT_PATH = os.path.join(os.getcwd(), self.report_dir)

            # Recupera la variable etiqueta para entrenar el modelo
            self.target_col = config.model_data.target_data
            
            logging.info("train data read successfully-->path: "+self.train_data)
            self.train = pd.read_csv(self.train_data, sep=",")
            logging.info("train data read successfully")
            self.test = pd.read_csv(self.test_data, sep=",")
            logging.info("test data read successfully")
            
            
            logging.info("model training started")
            self.criterion = config.estimators.RandomForestRegressor.params.criterion
            self.max_depth = config.estimators.RandomForestRegressor.params.max_depth
            self.min_sample_leaf = config.estimators.RandomForestRegressor.params.min_sample_leaf
            self.n_estimators = config.estimators.RandomForestRegressor.params.n_estimators
            self.min_sample_split = config.estimators.RandomForestRegressor.params.min_sample_split
            self.oob_score = config.estimators.RandomForestRegressor.params.oob_score

            # Cargar los datos
            self.x_train, self.x_test = self.train.drop(
                self.target_col, axis=1), self.test.drop(self.target_col, axis=1)
            self.y_train, self.y_test = self.train[self.target_col], self.test[self.target_col]
          
            # Definir modelo base
            rf = RandomForestRegressor()
            rf.fit(self.x_train, self.y_train)

            #Espacio de búsqueda
            distributions = { 
                "n_estimators": [5,10,20,40,80],
                "criterion": ["absolute_error", 'friedman_mse', 'poisson', 'squared_error'],
                "max_depth": [2,5,10],
                "min_samples_split": [2,4,8,12],
                "min_samples_leaf": [2,4,8,10]
            }

            # Ejecuta RandomizedSearchCV para encontrar los mejores hiperparametros            
            RCV = RandomizedSearchCV(
                estimator=rf,
                param_distributions= distributions,
                n_iter= config.RandomizedSearchCV.n_iter, 
                scoring= config.RandomizedSearchCV.scoring, 
                cv= config.RandomizedSearchCV.cv,
                verbose= config.RandomizedSearchCV.verbose,
                random_state= config.RandomizedSearchCV.random_state,
                n_jobs= config.RandomizedSearchCV.n_jobs,
                return_train_score= config.RandomizedSearchCV.return_train_score 
            )

            # Set tracking server uri for logging
            # No descomentes esta línea si vas a usar MLFlow de forma local
            # mlflow.set_tracking_uri(config.mlflow.tracking_uri)

            # Nombre del experimento
            exp_name = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            mlflow.set_experiment(exp_name)

            # Iniciar MLFlow
            with mlflow.start_run():

                # Ejecutar busqueda de los hiperparametros
                rf1 = RCV.fit(self.x_train, self.y_train)
                
                #Evaluar
                y_pred = rf1.predict(self.x_test)
                logging.info("Model Trained on RandomizedSearchCV successfully")
                (r2, mse, rmse) = self.evaluation_metrics(self.y_test, y_pred)

                #Infer model signature (agregado)
                self.x_train = self.x_train.astype("float64")
                predictions = rf1.predict(self.x_train)
                signature = infer_signature(self.x_train, predictions)

                # Registrar hiperparametros optimos
                mlflow.log_params(RCV.best_params_)

                # Registrar métricas
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("train_score", rf.score(self.x_train, self.y_train))
                mlflow.log_metric("test_score", rf.score(self.x_test, self.y_test))

                #Verifica que el directorio 'models' exista
                os.makedirs(MODEL_PATH,exist_ok=True)
                self.model_path = os.path.join(MODEL_PATH,self.filename)
                joblib.dump(rf1, self.model_path)

                # Registrar modelo MLFlow
                # Con el objetivo de no saturar los recursos limitados de su máquina local, 
                # puede omitir el guardado del modelo en local o en el servidor MLflow 
                # dejando la siguiente línea comentada
                #mlflow.sklearn.log_model(rf1, config.mlflow.mlruns_path, signature=signature) 

                #Verifica que el directorio 'reports' exista
                os.makedirs(REPORT_PATH,exist_ok=True)
                scores_file = config.reports.scores 
                params_file = config.reports.params

                with open(scores_file, "w") as f:
                    scores = {
                        "rmse": rmse,
                        "r2 score": r2*100,
                        "mse": mse,
                        "train_score": rf.score(self.x_train, self.y_train),
                        "test_score": rf.score(self.x_test, self.y_test)
                    }
                    json.dump(scores, f, indent=4)
                logging.info("scores written to file")
                
                with open(params_file, "w") as f:
                    params = {
                        "best params": RCV.best_params_,
                        "criterion": self.criterion,
                        "n_estimators": self.n_estimators,
                        "max_depth": self.max_depth,
                        "min_sample_leaf": self.min_sample_leaf,
                        "min_sample_split": self.min_sample_split,
                        "oob_score": self.oob_score
                    }
                    json.dump(params, f, indent=4)
                
            
        except Exception as e:
            logging.info("Exception occured in 'train_evaluate' function"+str(e))
            logging.info("train_evaluate function reported error in the function")
            raise AppException(e, sys) from e


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="model_eng", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    TrainEvaluate(cfg).model_eval(cfg)

if __name__ == "__main__":
    main()
