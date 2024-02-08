import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import  ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artefacts", "train.csv")
    test_data_path: str = os.path.join("artefacts", "test.csv")
    raw_data_path: str = os.path.join("artefacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion file")
        try:
            df = pd.read_csv("../../notebook/data/StudentsPerformance.csv")
            df.rename(columns={'math score': 'math_score',
                               'reading score': 'reading_score',
                               'writing score': 'writing_score',
                               'race/ethnicity': 'race_ethnicity',
                               'parental level of education': 'parental_level_of_education',
                               'test preparation course': 'test_preparation_course'},
                      inplace=True)
            logging.info("Loaded the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=27)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("data ingestion completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    tr_d, te_d = obj.initiate_data_ingestion()
    data_tran = DataTransformation()
    train_d, test_d, _ = data_tran.initiate_data_transform(tr_d,te_d)
    model_train = ModelTrainer()
    n, s = model_train.initiate_model_training(train_d,test_d)
    print(n, s)
