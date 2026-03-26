import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Path where the best model pickle will be saved."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains multiple regressors (with hyper-parameter tuning),
    picks the best one, builds an ensemble of the top-3 boosting
    models (CatBoost, XGBoost, AdaBoost), and saves whichever
    scores higher — the single best or the ensemble.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Args:
            train_array: np.ndarray (features + target as last column)
            test_array:  np.ndarray (features + target as last column)

        Returns:
            best R2 score (float)
        """
        try:
            logging.info("Splitting features and target from train/test arrays …")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test   = test_array[:, :-1],  test_array[:, -1]

            # ───────────────────────────────────────────────
            # 1. Define candidate models
            # ───────────────────────────────────────────────
            models = {
                "Linear Regression":     LinearRegression(),
                "Decision Tree":         DecisionTreeRegressor(),
                "Random Forest":         RandomForestRegressor(random_state=42),
                "K-Neighbors":           KNeighborsRegressor(),
                "Gradient Boosting":     GradientBoostingRegressor(random_state=42),
                "AdaBoost":              AdaBoostRegressor(random_state=42),
                "XGBoost":               XGBRegressor(random_state=42, verbosity=0),
                "CatBoost":              CatBoostRegressor(verbose=0, random_state=42),
            }

            # ───────────────────────────────────────────────
            # 2. Hyper-parameter grids (RandomizedSearchCV, 3-fold)
            # ───────────────────────────────────────────────
            params = {
                "Linear Regression": {},        # no tuning needed

                "Decision Tree": {
                    "max_depth":      [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                },

                "Random Forest": {
                    "n_estimators":   [50, 100, 200],
                    "max_depth":      [5, 10, None],
                    "min_samples_split": [2, 5],
                },

                "K-Neighbors": {
                    "n_neighbors":    [3, 5, 7, 9],
                    "weights":        ["uniform", "distance"],
                },

                "Gradient Boosting": {
                    "n_estimators":   [100, 200],
                    "learning_rate":  [0.05, 0.1, 0.2],
                    "max_depth":      [3, 5],
                    "subsample":      [0.8, 1.0],
                },

                "AdaBoost": {
                    "n_estimators":   [50, 100, 200],
                    "learning_rate":  [0.01, 0.05, 0.1, 0.5, 1.0],
                },

                "XGBoost": {
                    "n_estimators":   [100, 200],
                    "learning_rate":  [0.05, 0.1, 0.2],
                    "max_depth":      [3, 5, 7],
                    "subsample":      [0.8, 1.0],
                },

                "CatBoost": {
                    "iterations":     [100, 200, 300],
                    "learning_rate":  [0.01, 0.05, 0.1],
                    "depth":          [4, 6, 8],
                },
            }

            # ───────────────────────────────────────────────
            # 3. Train + tune all models
            # ───────────────────────────────────────────────
            logging.info("Starting model training with hyper-parameter tuning …")
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,   params=params,
            )

            # ───────────────────────────────────────────────
            # 4. Pick the single best model
            # ───────────────────────────────────────────────
            best_model_name  = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model       = models[best_model_name]

            logging.info(
                f"Best individual model: {best_model_name} "
                f"(R2 = {best_model_score:.4f})"
            )

            if best_model_score < 0.6:
                raise CustomException(
                    "No model achieved R2 >= 0.6. Model quality insufficient.", sys
                )

            # ───────────────────────────────────────────────
            # 5. Ensemble: VotingRegressor of the 3 boosting
            #    models (CatBoost, XGBoost, AdaBoost)
            # ───────────────────────────────────────────────
            logging.info("Building VotingRegressor ensemble (CatBoost + XGBoost + AdaBoost) …")
            ensemble = VotingRegressor(
                estimators=[
                    ("catboost",  models["CatBoost"]),
                    ("xgboost",   models["XGBoost"]),
                    ("adaboost",  models["AdaBoost"]),
                ],
            )
            ensemble.fit(X_train, y_train)
            ensemble_r2 = r2_score(y_test, ensemble.predict(X_test))
            logging.info(f"Ensemble VotingRegressor R2: {ensemble_r2:.4f}")

            # ───────────────────────────────────────────────
            # 6. Save the better of (best single vs ensemble)
            # ───────────────────────────────────────────────
            if ensemble_r2 > best_model_score:
                final_model = ensemble
                final_score = ensemble_r2
                final_name  = "Ensemble (CatBoost + XGBoost + AdaBoost)"
            else:
                final_model = best_model
                final_score = best_model_score
                final_name  = best_model_name

            logging.info(f"Final model selected → {final_name} (R2 = {final_score:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model,
            )
            logging.info(
                f"Model saved at: {self.model_trainer_config.trained_model_file_path}"
            )

            # ───────────────────────────────────────────────
            # 7. Print summary
            # ───────────────────────────────────────────────
            print("\n" + "=" * 60)
            print("MODEL TRAINING REPORT")
            print("=" * 60)
            for name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name:<25s}  R2 = {score:.4f}")
            print(f"  {'Ensemble (Boost trio)':<25s}  R2 = {ensemble_r2:.4f}")
            print("-" * 60)
            print(f"  ★ Final model saved: {final_name} (R2 = {final_score:.4f})")
            print("=" * 60 + "\n")

            return final_name, final_score

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------------------------------------------------
# Quick smoke-test — run directly:
#   python -m src.components.model_training
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from src.components.data_ingection import DataIngestion
    from src.components.data_processing import DataTransformation

    # Step 1: Ingest
    di = DataIngestion()
    train_path, test_path = di.initiate_data_ingestion()

    # Step 2: Transform
    dt = DataTransformation()
    train_arr, test_arr, _ = dt.initiate_data_transformation(train_path, test_path)

    # Step 3: Train
    mt = ModelTrainer()
    best_model_name, best_r2 = mt.initiate_model_trainer(train_arr, test_arr)
    print(f"Best Model: {best_model_name} | R2 Score: {best_r2:.4f}")
