import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Paths where the preprocessor pickle will be saved."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Builds and applies a preprocessing pipeline that:
      - Imputes missing values (median for numeric, most-frequent for categorical)
      - Scales numeric features with StandardScaler
      - Encodes categorical features with OneHotEncoder
    Saves the fitted preprocessor as a .pkl in the artifacts directory.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_data_transformer_object(self):
        """
        Construct and return the ColumnTransformer preprocessor.

        Numerical pipeline  : SimpleImputer(median)  → StandardScaler
        Categorical pipeline: SimpleImputer(most_frequent) → OneHotEncoder → StandardScaler
        """
        try:
            # ----- Column definitions (student performance dataset) -----
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # ----- Numeric pipeline -----
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            # ----- Categorical pipeline -----
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            logging.info(f"Numerical columns   : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            # ----- Combined preprocessor -----
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train/test CSVs, apply the preprocessing pipeline, save the
        fitted preprocessor to disk, and return transformed arrays.

        Args:
            train_path: Path to train.csv
            test_path:  Path to test.csv

        Returns:
            train_arr      : np.ndarray — transformed training data (features + target)
            test_arr       : np.ndarray — transformed test data    (features + target)
            preprocessor_path: str      — path to the saved preprocessor.pkl
        """
        try:
            # ---------- 1. Load data ----------
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            # ---------- 2. Separate features & target ----------
            target_column = "math_score"
            drop_columns  = [target_column]

            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column]

            logging.info("Separated input features and target column.")

            # ---------- 3. Build & fit the preprocessor ----------
            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Preprocessor object obtained. Fitting on training data …")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Preprocessing applied to training and test sets.")

            # ---------- 4. Combine features with target ----------
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(
                f"Transformed train shape: {train_arr.shape} | "
                f"test shape: {test_arr.shape}"
            )

            # ---------- 5. Save preprocessor pickle ----------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )
            logging.info(
                f"Preprocessor saved at: "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------------------------------------------------
# Quick smoke-test — run directly:
#   python -m src.components.data_processing
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from src.components.data_ingection import DataIngestion

    # Step 1: Ingest (creates artifacts/train.csv & artifacts/test.csv)
    di = DataIngestion()
    train_path, test_path = di.initiate_data_ingestion()

    # Step 2: Transform
    dt = DataTransformation()
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(
        train_path, test_path
    )

    print(f"Train array shape : {train_arr.shape}")
    print(f"Test  array shape : {test_arr.shape}")
    print(f"Preprocessor saved: {preprocessor_path}")
