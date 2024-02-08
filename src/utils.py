import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def eval_models(x_train, x_test, y_train, y_test, models):
    report = {}
    r = {}
    for i in models:
        model = models[i]
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        report[i] = test_score
    x = list(report.keys())
    y = list(report.values())
    z = np.argsort(y,)
    r = {x[i]: y[i] for i in z}
    return r
