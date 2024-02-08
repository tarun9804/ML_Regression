import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preproc_obj_file_path = os.path.join('artefacts', 'preproc.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            num_col = ['writing_score', 'reading_score']
            cat_col = ['gender',
                       'race_ethnicity',
                       'parental_level_of_education',
                       'lunch',
                       'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[("impute", SimpleImputer(strategy="median")),
                       ("scalar", StandardScaler())]
            )
            cat_pipeline = Pipeline(
                steps=[("impute", SimpleImputer(strategy="most_frequent")),
                       ("OnehotEncoder", OneHotEncoder()),
                       ("scalar", StandardScaler(with_mean=False))
                       ]
            )
            logging.info("Categorical and Numerical col encoding done")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col)
                    #('Onehotencoder', OneHotEncoder(), cat_col)
                    #('StandardScalar', StandardScaler(), num_col)
                ]
            )
            logging.info("preprocessor returned")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading train and test data done")

            preproc_obj = self.get_data_transformation_obj()
            target_col = "math_score"
            num_col = ['writing_score', 'reading_score']

            input_train_df = train_df.drop(columns=[target_col])
            target_train_df = train_df[target_col]

            input_test_df = test_df.drop(columns=[target_col])
            target_test_df = test_df[target_col]
            input_train_array = preproc_obj.fit_transform(input_train_df)
            input_test_array = preproc_obj.transform(input_test_df)

            train_arr = np.c_[input_train_array,np.array(target_train_df)]
            test_arr = np.c_[input_test_array,np.array(target_test_df)]

            save_obj(file_path=self.data_transformation_config.preproc_obj_file_path,
                     obj=preproc_obj)
            logging.info("preproc pkl saved")

            return train_arr,test_arr,self.data_transformation_config.preproc_obj_file_path
        except Exception as e:
            raise CustomException(e,sys)