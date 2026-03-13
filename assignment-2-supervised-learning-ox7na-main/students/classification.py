"""
Classification models for heart disease dataset.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_grid(X_train, y_train):
    """
    Train logistic regression with grid search.

    Returns
    -------
    GridSearchCV
        Fitted grid search object
    """
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    model = LogisticRegression(max_iter=5000, random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def train_knn_grid(X_train, y_train):
    """
    Train k-NN with grid search.

    Returns
    -------
    GridSearchCV
        Fitted grid search object
    """
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean"]
    }

    model = KNeighborsClassifier()

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def get_best_logistic_regression(X_train, y_train):
    """
    Fit logistic regression grid search and return best model info.

    Returns
    -------
    dict
        Dictionary with best_model, best_params, best_score, and grid_search
    """
    grid = train_logistic_regression_grid(X_train, y_train)

    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "grid_search": grid
    }


def get_best_knn(X_train, y_train):
    """
    Fit k-NN grid search and return best model info.

    Returns
    -------
    dict
        Dictionary with best_model, best_params, best_score, and grid_search
    """
    grid = train_knn_grid(X_train, y_train)

    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "grid_search": grid
    }
