import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Starting prediction pipeline")

            preprocessor_path = r"D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\column_processor.pkl"
            model_path = r"D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\model.pkl"
            label_encoder_path = r"D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\label_encoder.pkl"

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            label_encoder = load_object(label_encoder_path)

            # ✅ Convert all numeric columns to float64 to prevent casting issues
            numeric_cols = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps']
            for col in numeric_cols:
                if col in features.columns:
                    features[col] = features[col].astype('float64')

            logging.info("Preprocessing input features")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            preds = model.predict(data_scaled)

            logging.info("Decoding labels back to original classes")
            decoded_preds = label_encoder.inverse_transform(preds.astype(int))

            return decoded_preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, age, sleep_duration, physical_activity_level, stress_level,
                 daily_steps, gender, occupation, bmi_category, sleep_disorder):
        self.age = age
        self.sleep_duration = sleep_duration
        self.physical_activity_level = physical_activity_level
        self.stress_level = stress_level
        self.daily_steps = daily_steps
        self.gender = gender
        self.occupation = occupation
        self.bmi_category = bmi_category
        self.sleep_disorder = sleep_disorder

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Age": [self.age],
                "Sleep Duration": [self.sleep_duration],
                "Physical Activity Level": [self.physical_activity_level],
                "Stress Level": [self.stress_level],
                "Daily Steps": [self.daily_steps],
                "Gender": [self.gender],
                "Occupation": [self.occupation],
                "BMI Category": [self.bmi_category],
                "Sleep Disorder": [self.sleep_disorder]
            }

            df = pd.DataFrame(data_dict)

            # ✅ Ensure numeric columns are float64 here too
            numeric_cols = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Daily Steps']
            df[numeric_cols] = df[numeric_cols].astype('float64')

            logging.info("Custom data converted to DataFrame successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)
