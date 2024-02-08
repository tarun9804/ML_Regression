import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import eval_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artefacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("train test split")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-NearestNeighbour": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }
            model_report: dict = eval_models(x_train, x_test, y_train, y_test, models)
            temp = list(model_report.items())[-1]
            best_model_score = temp[1]
            best_model_name = temp[0]
            logging.info(f"best model {best_model_name} found")

            save_obj(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model_name
            )
            return best_model_name,best_model_score

        except Exception as e:
            pass
