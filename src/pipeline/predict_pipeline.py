import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:

            model_path = 'src/components/artefacts/model.pkl'
            pre_proc_path = 'src/components/artefacts/preproc.pkl'
            model = load_object(file_path=model_path)
            pre_proc = load_object(file_path=pre_proc_path)
            data_scaled = pre_proc.transform(feature)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            custome_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(custome_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)



