import os
import pandas as pd
import numpy as np
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, Y_train, X_test, Y_test, models, param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            logging.info(f"Training {model_name} with hyperparameter tuning")
            
            gs = GridSearchCV(model, para, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train, Y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(Y_train, y_train_pred)
            test_model_score = accuracy_score(Y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
            logging.info(f"{model_name} - Best Params: {gs.best_params_}")
            logging.info(f"{model_name} - Train Accuracy: {train_model_score:.2f}, Test Accuracy: {test_model_score:.2f}")
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)