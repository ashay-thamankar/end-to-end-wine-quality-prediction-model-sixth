from flask import Flask , render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.components.model_prediction import ModelPrediction
from mlProject.config.configuration import ConfigurationManager


app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def training():
    os.system('python main.py')
    return render_template('success.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    config = ConfigurationManager()
    model_predict_config = config.get_model_prediction_config()
    model_prediction = ModelPrediction(config = model_predict_config)

    if request.method == "POST":
        try:
            
            data = model_prediction.get_data_as_data_frame(
            fixed_acidity = float(request.form.get("fixed_acidity")),
            volatile_acidity = float(request.form.get("volatile_acidity")),
            citric_acid = float(request.form.get("citric_acid")),
            residual_sugar = float(request.form.get("residual_sugar")),
            chlorides = float(request.form.get("chlorides")),
            free_sulfur_dioxide = float(request.form.get("free_sulfur_dioxide")),
            total_sulfur_dioxide = float(request.form.get("total_sulfur_dioxide")),
            density = float(request.form.get("density")),
            pH = float(request.form.get("pH")),
            sulphates = float(request.form.get("sulphates")),
            alcohol = float(request.form.get("alcohol"))
        )

            pred = model_prediction.predict_data(data)

            return render_template('results.html', prediction = str(round(pred[0],3)))
        except Exception as e:
            print(f"The Exception message is : ",e)
            return render_template('error.html', error_message = e)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(host='0.0.0.0', port=8080)