"""
Regression models for heart disease dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


def train_elasticnet_grid(X_train, y_train, l1_ratios=None, alphas=None):
    """
    Train ElasticNet models over a grid of l1_ratio and alpha values.

    Returns
    -------
    pd.DataFrame
        DataFrame with l1_ratio, alpha, r2_score, model, and rank
    """
    if l1_ratios is None:
        l1_ratios = [0.3, 0.5, 0.7]
    if alphas is None:
        alphas = [0.01, 0.1, 1.0]

    param_grid = {
        "l1_ratio": l1_ratios,
        "alpha": alphas
    }

    grid = GridSearchCV(
        estimator=ElasticNet(max_iter=10000, random_state=42),
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    results_df = pd.DataFrame({
        "l1_ratio": grid.cv_results_["param_l1_ratio"].astype(float),
        "alpha": grid.cv_results_["param_alpha"].astype(float),
        "r2_score": grid.cv_results_["mean_test_score"].astype(float),
        "rank": grid.cv_results_["rank_test_score"].astype(int)
    })

    # Store fitted model objects for each parameter combination
    models = []
    for _, row in results_df.iterrows():
        model = ElasticNet(
            l1_ratio=float(row["l1_ratio"]),
            alpha=float(row["alpha"]),
            max_iter=10000,
            random_state=42
        )
        model.fit(X_train, y_train)
        models.append(model)

    results_df["model"] = models

    return results_df


def create_r2_heatmap(results_df):
    """
    Create heatmap of R² scores across ElasticNet hyperparameters.
    """
    pivot_table = results_df.pivot(index="l1_ratio", columns="alpha", values="r2_score")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_table.values, aspect="auto")

    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels([str(x) for x in pivot_table.columns])
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels([str(y) for y in pivot_table.index])

    ax.set_xlabel("alpha")
    ax.set_ylabel("l1_ratio")
    ax.set_title("ElasticNet R² Heatmap")

    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            ax.text(j, i, f"{pivot_table.iloc[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Mean CV R²")
    plt.tight_layout()

    return fig


def get_best_elasticnet_model(X_train, y_train, l1_ratios=None, alphas=None):
    """
    Fit ElasticNet grid search and return best model information.
    """
    if l1_ratios is None:
        l1_ratios = [0.3, 0.5, 0.7]
    if alphas is None:
        alphas = [0.01, 0.1, 1.0]

    param_grid = {
        "l1_ratio": l1_ratios,
        "alpha": alphas
    }

    grid = GridSearchCV(
        estimator=ElasticNet(max_iter=10000, random_state=42),
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "grid_search": grid
    }