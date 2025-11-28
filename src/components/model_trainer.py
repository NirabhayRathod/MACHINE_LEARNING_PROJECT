import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

def initiate_model_training(train_array, test_array):
    try:
        logging.info("Splitting train and test arrays into features and targets")
        
        X_train = train_array[:, :-1]  # All columns except last
        Y_train = train_array[:, -1]   # Last column (SleepQuality)
        X_test = test_array[:, :-1]    # All columns except last  
        Y_test = test_array[:, -1]     # Last column (SleepQuality)

        logging.info(f"Training set shape: X_train {X_train.shape}, Y_train {Y_train.shape}")
        logging.info(f"Test set shape: X_test {X_test.shape}, Y_test {Y_test.shape}")
        logging.info(f"SleepQuality value counts in training set: {pd.Series(Y_train).value_counts().to_dict()}")

        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }

        params = {
            'Decision Tree': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [50, 100]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'newton-cg']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'XGBoost': {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            },
            'CatBoost': {
                'depth': [6, 8],
                'learning_rate': [0.05, 0.1],
                'iterations': [100, 200]
            },
            'AdaBoost': {
                'learning_rate': [0.1, 0.5],
                'n_estimators': [50, 100]
            }
        }

        logging.info("Evaluating models with provided hyperparameters")
        model_report = evaluate_models(
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            models=models, param=params
        )

        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        if best_model_score <= 0.6:
            raise CustomException("No sufficiently good model found")

        logging.info(f"Best model selected: {best_model_name} with Accuracy score: {best_model_score}")
        
        # Save the best model
        save_object(file_path=r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\model.pkl', obj=best_model)

        # Make predictions and calculate final accuracy
        predictions = best_model.predict(X_test)
        final_score = accuracy_score(Y_test, predictions)
        
        # Log classification report for detailed performance
        logging.info(f"Classification Report for {best_model_name}:\n{classification_report(Y_test, predictions)}")
        logging.info(f"SleepQuality distribution - Actual: {pd.Series(Y_test).value_counts().to_dict()}")
        logging.info(f"SleepQuality distribution - Predicted: {pd.Series(predictions).value_counts().to_dict()}")

        return final_score

    except Exception as e:
        raise CustomException(e, sys)