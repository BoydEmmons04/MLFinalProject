# ---------------------------------------------------------------
# Authors: Carter Ward, Boyd Emmons
# Class  : CS 430 - 1
# Date   : 11/4/2025
#
# Project: ML Final Project – Census & Income Data
# Purpose: 
#   This project creates an Artificial Neural Network (ANN) and a 
#   Support Vector Machine (SVM) and compares their outputs for a 
#   Binary Classification problem. This classifies whether or not 
#   an individual's income exceeds $50K a year given training data.
#
#   The following code provides a structured pipeline for loading,
#   cleaning, preprocessing, and preparing the dataset before model
#   development. Each section includes notes describing its role and 
#   reasonable methods to implement.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Imports (minimal project skeleton)
# ---------------------------------------------------------------
import logging
from pathlib import Path
from typing import Tuple, Optional, Any

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

# Optional deep learning import (may not be installed in all envs)
try:
	import tensorflow as tf
	from tensorflow import keras
except Exception:
	tf = None
	keras = None


# ---------------------------------------------------------------
# File Paths / Constants
# ---------------------------------------------------------------
ROOT = Path(__file__).parent
TRAIN_FILE: Path = ROOT / "adult.data"
TEST_FILE: Path = ROOT / "adult.test"
NAMES_FILE: Path = ROOT / "adult.names"


# ---------------------------------------------------------------
# Pipeline function stubs (minimal)
# Each function below is a placeholder with a
# NotImplementedError to indicate where full implementations
# should be added. Keep changes here minimal per project start.
# ---------------------------------------------------------------


def load_dataset(path: Path) -> pd.DataFrame:
	"""Load dataset from `path` into a pandas DataFrame.

	TODO: implement parsing, column names from `NAMES_FILE`, and
	any header cleaning required for `adult.test`.
	"""
	raise NotImplementedError("load_dataset is a placeholder; implement parsing logic")


def explore_dataset(df: pd.DataFrame) -> None:
	"""Perform lightweight EDA (shapes, dtypes, missing counts).

	TODO: log/print helpful summaries and visualizations.
	"""
	raise NotImplementedError("explore_dataset is a placeholder")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Identify and impute or clean missing values.

	TODO: choose imputation strategies (mode for categorical, etc.).
	"""
	raise NotImplementedError("handle_missing_values is a placeholder")


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
	"""Encode categorical columns and return (df_encoded, encoder).

	TODO: implement OneHotEncoding or pd.get_dummies and return encoder
	for later reuse on test set.
	"""
	raise NotImplementedError("encode_categorical is a placeholder")


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
	"""Scale numeric features and return (X_scaled, scaler).

	TODO: implement StandardScaler or MinMaxScaler as appropriate.
	"""
	raise NotImplementedError("scale_features is a placeholder")


def split_features_labels(df: pd.DataFrame) -> Tuple[Any, Any]:
	"""Separate features (X) and labels (y) from preprocessed df.

	TODO: map target to binary (0/1) and align columns.
	"""
	raise NotImplementedError("split_features_labels is a placeholder")


def save_preprocessed(df: pd.DataFrame, out_path: Path) -> None:
	"""Save cleaned/preprocessed DataFrame to CSV for reuse.

	Keeps IO centralized and reproducible.
	"""
	# Minimal placeholder: actual save should call df.to_csv(out_path)
	raise NotImplementedError("save_preprocessed is a placeholder")


def build_ann_model(input_shape: int) -> Any:
	"""Return an uncompiled ANN model (keras) or None if keras missing.

	TODO: implement architecture, compile, and return keras.Model.
	"""
	if keras is None:
		logging.warning("Keras not available in this environment")
		return None
	raise NotImplementedError("build_ann_model is a placeholder")


def build_svm_model() -> SVC:
	"""Return an untrained SVM classifier instance.

	TODO: set kernel, C, class_weight, and other hyperparameters.
	"""
	return SVC()


def evaluate_models(models: dict, X_test: Any, y_test: Any) -> dict:
	"""Evaluate trained models on test data and return metrics dict.

	TODO: compute accuracy, precision, recall, f1, and confusion matrix.
	"""
	raise NotImplementedError("evaluate_models is a placeholder")


def main() -> None:
	"""Main entry for the pipeline skeleton.

	This function currently only outlines the intended calls. Fill in
	the implementations above to run the full pipeline.
	"""
	logging.basicConfig(level=logging.INFO)
	logging.info("Project skeleton initialized. Fill in implementations in each function.")
	print("Project skeleton initialized. Open this file and implement functions.")


if __name__ == "__main__":
	main()

