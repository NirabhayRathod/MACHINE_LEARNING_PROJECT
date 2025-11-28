import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

def initiate_data_ingestion():
    try:
        data = pd.read_csv(r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\cleaned_dataset.csv')
        logging.info('data reading completed from -> "SleepPattern_BalancedRaw.csv"')
        
        train, test = train_test_split(data, random_state=42, test_size=0.2)
        logging.info('data splitting completed')
         
        train.to_csv(r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\train.csv', index=False)
        test.to_csv(r'D:\SushilPal\MACHINE_LEARNING_PROJECT\artifacts\test.csv', index=False)
        
        return train, test

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    train_data, test_data = initiate_data_ingestion()

    from data_transformation import initiate_data_transformation
    from model_trainer import initiate_model_training

    train_arr, test_arr, column_processor_path, label_encoder_path = initiate_data_transformation(train_data, test_data)

    score = initiate_model_training(train_arr, test_arr)
    print(f"Our best model score: {score}")