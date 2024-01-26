import pandas as pd
from sklearn.model_selection import train_test_split
from mlProject.config.configuration import DataTransformationConfig
from mlProject import *
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from mlProject.utils.common import save_object, load_object
from pathlib import Path


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation_object(self):
        ''' This function is responsible for data transformation'''
        try:
            numerical_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
            categorical_columns = []
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns : {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise e
        
    def initiate_data_transformation(self):
        try:
            data = pd.read_csv(self.config.data_path)

            train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

            train_df.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
            test_df.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

 
            logging.info("Read Train and Test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformation_object()

            save_object(file_path=self.config.preprocessor_path, obj=preprocessor_obj)
            logger.info(f'preprocessor object has saved at {self.config.preprocessor_path}')

            target_column_name = 'quality'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on trianing and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(file_path= Path(self.config.train_array_path), obj= train_arr)
            save_object(file_path= Path(self.config.test_array_path), obj=test_arr)

            logging.info('train and test array saved.')

        except Exception as e:
            raise e