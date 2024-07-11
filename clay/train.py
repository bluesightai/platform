from typing import Any, Dict, Tuple

import numpy as np
from numpy._typing import NDArray
from sklearn import svm


def train_classification(embeddings: NDArray, labels: NDArray) -> Tuple[svm.SVC, Dict[str, Any]]:

    # from xgboost import XGBClassifier
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #    'n_estimators': [50, 100, 150],
    #    'learning_rate': [0.01, 0.1, 0.2],
    #    'max_depth': [3, 5, 7]
    # }
    # xgb = XGBClassifier()
    # grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
    # grid_search.fit(embeddings, labels)
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print(f"Best parameters found: {best_params}")

    clf = svm.SVC()
    clf.fit(embeddings, labels)

    details = {k: v for k, v in clf.__dict__.items() if k not in ["support_vectors_"]}
    details = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in details.items()}

    return clf, details


def predict_classification(clf: svm.SVC, embeddings: NDArray) -> NDArray:
    labels = clf.predict(embeddings)
    return labels
