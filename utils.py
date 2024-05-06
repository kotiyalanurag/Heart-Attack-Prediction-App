import os
import yaml
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

def read_yaml(file: yaml):
    """ A function that reads the contents of a yaml file

    Args:
        file (yaml): a yaml file that contains model parameters

    Returns:
        _type_: a dictory containing model parameters
    """
    with open(file, "r") as f:
        content = yaml.safe_load(f)
        
    return content

def plot_histogram(data: pd.DataFrame, col_names: list, col_labels: list):
    
    """ Creates a histogram of given features in a dataset

    Args:
        data (pd.DataFrame): dataset
        col_names (list): column names in the dataset
        col_labels (list): actual column labels
    """
    sns.set_style("darkgrid")
    
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (16, 8))
    axes = axes.flatten()

    for i, (col, lab) in enumerate(zip(col_names, col_labels)):
        sns.histplot(data[col], kde = True, ax = axes[i], bins = 20)
        axes[i].set_xlabel(lab)
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"'{col}' distribution in dataset")

    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_barchart(data: pd.DataFrame, col_names: list, col_labels: list):
    
    """ Creates a barchart of given features in a dataset wrt output labels

    Args:
        data (pd.DataFrame): dataset
        col_names (list): column names in the dataset
        col_labels (list): actual column labels
    """
    fig, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (16, 16))
    axes = axes.flatten()

    for i, (col, lab) in enumerate(zip(col_names, col_labels)):
        sns.countplot(data = data, x = col, hue = 'label', ax = axes[i], palette = "rocket")
        axes[i].set_xlabel(lab)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"'{col}' count in dataset")

    plt.tight_layout()
    plt.show()

def plot_correlation(data: pd.DataFrame):
    
    """ Creates a heatmap of correlation between different features in a dataset

    Args:
        data (pd.DataFrame): dataset
    """
    plt.figure(figsize = (16, 8))

    corr_mat = data.corr(numeric_only = True)
    sns.heatmap(corr_mat, cmap = "YlGnBu", annot = True)

    plt.title("Feature Correlation Heatmap")
    plt.show()
    
def find_best_params(name, features, labels, hyperparameters):
    
    """ A functions that returns best parameters of logistic regression or svm classifier

    Args:
        name (_type_): model name i.e., logistic regression (Logistic) or svm (SVM)
        features (_type_): training features (X_train)
        labels (_type_): training labels (y_train)
        hyperparameters (_type_): a dictionary containing model hyperparameters

    Returns:
        _type_: best model hyperparameters found by gridsearch cross validation
    """
    if name == "Logistic":
        model = LogisticRegression()
        grid_search = GridSearchCV(estimator = model, 
                           param_grid = hyperparameters, 
                           scoring = 'f1',
                           cv = 5, verbose = 1)

        grid_search.fit(features, labels)
    
    elif name == "SVM":
        model = SVC()
        grid_search = GridSearchCV(estimator = model, 
                           param_grid = hyperparameters, 
                           scoring = 'f1',
                           cv = 5, verbose = 1)

        grid_search.fit(features, labels.values.ravel())
        
    else:
        return 0
        
    return grid_search.best_params_

def find_best_k(features, labels, test_features, test_labels):
    
    """ A functions that creates a graph that shows best k using elbow method

    Args:
        features (_type_): train features
        labels (_type_): train labels
        test_features (_type_): test features
        test_labels (_type_): test labels
    """
    error_rate = []

    for i in range(1, 10):
    
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(features, labels)
        pred_i = knn.predict(test_features)
        
        error_rate.append(np.mean(pred_i != test_labels))

    plt.figure(figsize=(16, 8))
    plt.plot(range(1, 10), error_rate, color='blue', 
            linestyle='dashed', marker='o', 
            markerfacecolor='red', markersize=10)
    plt.title('Error rate vs. k-nearest neighbors')
    plt.xlabel('K-neighbors')
    plt.ylabel('Error rate')
    plt.show()

def save_model(model, name):
    
    """ A function that saves models as pickle files in 

    Args:
        model (_type_): an sklearn model
        name (_type_): a name to save the model
    """
    dir_path = os.path.join(os.getcwd(), 'models') 
    file_path = os.path.join(dir_path, name)   
    
    with open(f"{file_path}", 'wb') as f:
        pickle.dump(model, f)
        
def load_model(name):
    
    """ A function that loads trained models

    Args:
        name (_type_): name of the model to be loaded

    Returns:
        _type_: loaded model
    """
    dir_path = os.path.join(os.getcwd(), 'models') 
    file_path = os.path.join(dir_path, name)
    
    with open(f"{file_path}", "rb") as f:
        model = pickle.load(f)
        
    return model