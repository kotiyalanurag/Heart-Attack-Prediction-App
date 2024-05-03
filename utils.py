import yaml
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report
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