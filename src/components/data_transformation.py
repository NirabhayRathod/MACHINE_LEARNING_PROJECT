import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# FIX: Import MinMaxScaler instead of StandardScaler to avoid type casting issues
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

def get_data_trans_obj():
    try:
        num_columns = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps']
        cat_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

        # FIX: Use MinMaxScaler instead of StandardScaler - no type casting issues
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())  # CHANGED: MinMaxScaler handles mixed types better
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        logging.info(f"Numerical columns: {num_columns}")
        logging.info(f"Categorical columns: {cat_columns}")

        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, num_columns),
            ('cat_pipeline', cat_pipeline, cat_columns)
        ])

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_transformation(train, test):
    try:
        train_df = train
        test_df = test
        logging.info("Read train and test data")

        # FIX: Convert to float64 for ALL numerical columns to avoid casting issues
        numerical_cols = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps']
        
        for col in numerical_cols:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype('float64')
                test_df[col] = test_df[col].astype('float64')

        preprocessor = get_data_trans_obj()
        target_column = 'SleepQuality'

        input_feature_train_df = train_df.drop(columns=[target_column])
        target_feature_train_df = train_df[target_column]

        input_feature_test_df = test_df.drop(columns=[target_column])
        target_feature_test_df = test_df[target_column]

        logging.info("Applying preprocessing object on training and testing dataframes")

        input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessor.transform(input_feature_test_df)

        # Apply LabelEncoder to target feature
        label_encoder = LabelEncoder()
        target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
        target_feature_test_encoded = label_encoder.transform(target_feature_test_df)

        logging.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
        test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]

        # Save both preprocessor and label encoder
        save_object(file_path=r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\column_processor.pkl', obj=preprocessor)
        save_object(file_path=r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\label_encoder.pkl', obj=label_encoder)
        logging.info("Preprocessing object and label encoder saved")

        return train_arr, test_arr, r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\column_processor.pkl', r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\label_encoder.pkl'

    except Exception as e:
        raise CustomException(e, sys)