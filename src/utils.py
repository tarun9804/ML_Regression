import os
import sys
import dill
import joblib
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def eval_models(x_train, x_test, y_train, y_test, models, param, file_path):
    report = {}
    r = {}
    for i in models:
        model = models[i]
        para = param[i]
        print(f"running model {i}...")
        gs = GridSearchCV(model, para, cv=3)
        gs.fit(x_train, y_train)

        model.set_params(**gs.best_params_)

        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        report[i] = test_score
    x = list(report.keys())
    y = list(report.values())
    z = np.argsort(y)
    r = {x[i]: y[i] for i in z}
    '''
    best_model_name = x[z[-1]]
    model = models[best_model_name]
    para = param[best_model_name]
    print(f"running best model {best_model_name}...")
    gs = GridSearchCV(model, para, cv=3)
    gs.fit(x_train, y_train)
    model.set_params(**gs.best_params_)
    model.fit(x_train, y_train)
    save_obj(file_path,model)
    '''

    return r


def load_object(file_path):
    try:
        print(file_path)
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)


