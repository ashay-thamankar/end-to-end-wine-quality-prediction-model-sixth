from mlProject.config.configuration import ModelPredictionConfig
from pathlib import Path
from mlProject.utils.common import load_object, get_data_transformation_object
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mlProject.components.data_transformation import DataTransformation

class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config

    def get_data_as_data_frame(self, fixed_acidity: float, 
                 volatile_acidity: float, 
                 citric_acid: float, 
                 residual_sugar: float,
                 chlorides: float,
                 free_sulfur_dioxide: float,
                 total_sulfur_dioxide: float,
                 density: float,
                 pH: float,
                 sulphates: float,
                 alcohol: float):
        try:
            custom_data_input_dict = {
                "fixed_acidity": fixed_acidity,
                "volatile_acidity": volatile_acidity,
                "citric_acid": citric_acid,
                "residual_sugar": residual_sugar,
                "chlorides": chlorides,
                "free_sulfur_dioxide": free_sulfur_dioxide,
                "total_sulfur_dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol
            }
            print(custom_data_input_dict)
            input_array = np.array(list(custom_data_input_dict.values())).reshape(1, 11)
            print(input_array)
            return input_array
        except Exception as e:
            raise e

    def predict_data(self, features):
        try:
            print("before loading")
            model = load_object(file_path= Path(self.config.model_path))
            preprocessor = load_object(file_path=Path(self.config.preprocessor_path))
            best_parameters = load_object(file_path=Path(self.config.params_path))
            train_array = load_object(file_path= Path(self.config.train_array_path))
            print(f"Best params {best_parameters}")
            print(f"Train array : {train_array}, {train_array.shape}")
            print(f"features : {features}")

            train_x, train_y = (train_array[:,:-1], train_array[:,-1])
            print('after trainx and train_y')   
            
            preprocessor_obj = get_data_transformation_object()
            print('after object creation')
            print(train_x)
            # print(np.array(train_x))

            # train_x_array = preprocessor_obj.transform(train_x)
            print('after fit transfer')

            model.set_params(**best_parameters)
            model.fit(train_x, train_y)
            print('after fit')
            print("After loading")
            # print(features.values())
            # data_scaled = preprocessor_obj.transform(features)
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise e
        
    

    