# end-to-end-machine-learning-project-with-mlops

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update entity
5. Update the configuration manager in src config
6. Update the components 
7. Update the pipeline
8. Update the main.py
9. Update the app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/ashay-thamankar/end-to-end-machine-learning-project-with-mlops.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p venv python=3.8 -y
```

```bash
activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/ashaythamankar/end-to-end-machine-learning-project-with-mlops.mlflow \
MLFLOW_TRACKING_USERNAME=ashaythamankar \
MLFLOW_TRACKING_PASSWORD=eeec17ca9a903e7572f126b2e8f6d76eea2c4bd9 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/ashaythamankar/end-to-end-machine-learning-project-with-mlops.mlflow

export MLFLOW_TRACKING_USERNAME=ashaythamankar 

export MLFLOW_TRACKING_PASSWORD=eeec17ca9a903e7572f126b2e8f6d76eea2c4bd9

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 511882280103.dkr.ecr.ap-south-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = ap-south-1

    AWS_ECR_LOGIN_URI = demo>>  511882280103.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = mlproj




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model

# End-to-End Wine Quality Prediction Model 🍷

## Problem Statement
In this project, our goal is to predict the quality of wines based on various chemical and physical properties. The dataset contains 1599 different wines, each characterized by 12 distinct features, including density, pH level, and chemical composition.

## General ML Cycle Workflow 🔄
1. **Scope Project**: Define the objectives and scope of the project.
2. **Collect Data**: Gather and understand the dataset.
3. **Train Model**: Develop and train the machine learning model.
4. **Deploy in Production**: Deploy, monitor, and maintain the system.

## Project Workflow 🚀
### Development Stage
- **Data Ingestion**: Collect and load the dataset.
- **Data Transformation**: Process and clean the data.
- **Model Trainer**: Develop and train the machine learning model.
- **Model Evaluation**: Assess the model's performance.

### Deployment Stage
- **Dockerization**: Create a Docker image for deployment.
- **AWS Deployment**: Deploy the model on AWS.
- **Model Monitoring**: Implement monitoring for the deployed model.

## Project Deployment Demonstration 🌐
- Access the deployed model using the AWS EC2 instance.
- Input feature values to predict wine quality.

## Insights and Analysis 📊
- Gain insights from the dataset through statistical analysis.
- Iterative improvement through multiple model training sessions.

## Gratitude 🙏
Thank you for your time and consideration. Feedback and suggestions are welcome!



