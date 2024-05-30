<h1 align=center> Heart Attack Prediction

![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/sklearn-1.4.2-blue) ![](https://img.shields.io/badge/fastapi-0.111.0-blue) ![](https://img.shields.io/badge/docker-7.1.0-blue) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

<p align = left>Using a logistic regression model to predict a person's chance of suffering from a heart attack using their age, sex, resting blood pressure, cholestrol levels, blood sugar levels, maximum heart rates, and chest pain type.</p>

## Overview

The model is capable of predicting a "low chance" or "high chance" of a heart attack for a patient given the results of their clinic reports.

## Hyperparameters

Three models were trained for this project i.e., logistic regression, k nearest neighbors, and support vector machines. Their hyperparameters can be changed using the 'params.yaml' file.

```yaml
# logistic regression
lr:
  solver: ['lbfgs', 'liblinear', 'newton-cg']
  max_iter: [25, 50, 100]


# k-nearest neighbors
knn:
  neighbors: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# support vector machines
svm:
  C: [0.05, 0.1, 0.5, 1, 10]
  kernel: ['poly', 'rbf', 'sigmoid']
```
For hyperparamter tuning grid search cv was used with svm and logistic regression. To find an optimal k for KNN, the elbow plot method was used.

## How to train and use these models?

Use the jupyter notebook inside notebook directory. Latest versions of trained models will automatically be saved inside the models directory.

## How to run the fastapi app?

Just run the following script from your terminal

```python
$ uvicorn app:app
```
Go to the link displayed in terminal and open the swagger UI of fastapi. You'll see something like this.

<p align="center">
  <img src = assets/app.png max-width = 100% height = '295' />
</p>

## How to run the docker container?

Just build the container using the following script on terminal. Replace my-app-name with a name that you'd like for your image.

```python
$ docker build -t my-app-name .
```
And once the image is built just run a container using

```python
$ docker run -p 8000:8000 my-app-name
```
You'll see this on terminal. Just open the link and you'll find that the app is running inside a docker container.

<p align="center">
  <img src = assets/docker.png max-width = 100% height = '150' />
</p>

## Model Performance

The best logistic regression model had an accuracy of 84% which is pretty good given the small size of this dataset. The knn model and svm model had an accuracy close to 82% as well.

Using the fastapi swagger UI, we can feed the model input values to get a prediction. Under /predict click on try it out and feed the input values and hit execute. The result will be displayed below.

<p align="center">
  <img src = assets/inp.png max-width = 100% height = '210' />
</p>

<p align="center">
  <img src = assets/pred.png max-width = 100% height = '70' />
</p>

## Dataset

The dataset can be downloaded from [here.](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)