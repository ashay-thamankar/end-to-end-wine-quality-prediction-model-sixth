from mlProject.config.configuration import ModelPredictionConfig
from pathlib import Path
from mlProject.utils.common import load_object
import pandas as pd

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
            return_pd_data = pd.DataFrame([custom_data_input_dict])
            print(return_pd_data.values)
            # csv_data = pd.to_csv(return_pd_data)
            return return_pd_data
        except Exception as e:
            raise e

    def predict_data(self, features):
        try:
            model_path = Path(self.config.model_path)
            preprocessor_path = Path(self.config.preprocessor_path)
            best_params_path = Path(self.config.params_path)
            train_array_path = Path(self.config.train_array_path)
            print("before loading")
            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            best_parameters = load_object(file_path=best_params_path)
            train_array = load_object(file_path= train_array_path)
            train_x, train_y = (train_array[:,:-1], train_array[:,-1])
            model.set_params(**best_parameters)
            model.fit(train_x, train_y)
            print("After loading")
            preprocessor.fit(train_x)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise e
        
    

    