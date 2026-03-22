import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging


def save_object(file_path, obj):
    """Save a Python object to a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models, returning a report of R2 scores.

    Args:
        X_train: Training features
        y_train: Training target
        X_test:  Test features
        y_test:  Test target
        models:  dict of {"model_name": model_instance}

    Returns:
        dict of {"model_name": r2_score}
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[model_name] = test_r2

            logging.info(
                f"{model_name} — Train R2: {train_r2:.4f} | "
                f"Test R2: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
