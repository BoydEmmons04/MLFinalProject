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

# Canonical column names from adult.names
COLUMN_NAMES = [
	'age', 'workclass', 'fnlwgt', 'education', 'education-num',
	'marital-status', 'occupation', 'relationship', 'race', 'sex',
	'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]


# ---------------------------------------------------------------
# Internal helpers for dataset loading/cleaning
# ---------------------------------------------------------------
def _is_test_file(p: Path) -> bool:
	return p.name.lower().endswith("adult.test")


def _clean_test_labels(df: pd.DataFrame) -> pd.DataFrame:
	# adult.test labels include a trailing '.' (e.g., '>50K.'), strip it.
	if 'income' in df.columns and pd.api.types.is_object_dtype(df['income']):
		df['income'] = df['income'].str.strip().str.replace(r"\.$", "", regex=True)
	return df


# ---------------------------------------------------------------
# Pipeline function implementations / stubs
# ---------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
	"""Load the Adult (Census Income) dataset from a file path.

	Supports:
	  - adult.data  (training)
	  - adult.test  (testing; skips first header line and strips label periods)

	Behavior:
	  - Assigns canonical COLUMN_NAMES
	  - Treats '?' as missing (NaN)
	  - Trims leading spaces in fields
	"""
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")

	is_test = _is_test_file(path)
	read_kwargs = dict(
		header=None,
		names=COLUMN_NAMES,
		na_values=['?'],
		skipinitialspace=True
	)
	if is_test:
		# adult.test has a first line that's not data
		read_kwargs['skiprows'] = 1

	df = pd.read_csv(path, **read_kwargs)
	if is_test:
		df = _clean_test_labels(df)

	# Sanity check: expect exactly 15 columns
	if df.shape[1] != len(COLUMN_NAMES):
		raise ValueError(
			f"Unexpected column count in {path.name}: got {df.shape[1]}, expected {len(COLUMN_NAMES)}"
		)
	return df


def explore_dataset(df: pd.DataFrame) -> None:
	"""Lightweight EDA: shape, dtypes, missingness, class balance, and numeric summary."""
	logger = logging.getLogger("EDA")

	logger.info("Shape: %s", df.shape)
	logger.info("Dtypes:\n%s", df.dtypes)

	with pd.option_context('display.max_rows', 5, 'display.max_columns', 20):
		logger.info("Head:\n%s", df.head())

	# Missing values overview
	na_counts = df.isna().sum().sort_values(ascending=False)
	logger.info("Missing values per column (desc):\n%s", na_counts)

	# Target distribution (if present)
	if 'income' in df.columns:
		class_counts = df['income'].value_counts(dropna=False)
		class_ratio = (class_counts / class_counts.sum()).round(4)
		logger.info("Target distribution (counts):\n%s", class_counts.to_string())
		logger.info("Target distribution (proportions):\n%s", class_ratio.to_string())

	# Basic type split
	cat_cols = df.select_dtypes(include='object').columns.tolist()
	num_cols = df.select_dtypes(exclude='object').columns.tolist()
	logger.info("Numeric columns (%d): %s", len(num_cols), num_cols)
	logger.info("Categorical columns (%d): %s", len(cat_cols), cat_cols)

	# Descriptive statistics for numeric features
	if num_cols:
		with pd.option_context('display.max_rows', 100, 'display.max_columns', 20):
			logger.info("Numeric summary:\n%s", df[num_cols].describe().T)


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
	logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
	logging.info("Project skeleton initialized. Fill in implementations in each function.")

	# Example usage for the implemented parts:
	try:
		train_df = load_dataset(TRAIN_FILE)
		test_df = load_dataset(TEST_FILE)
		explore_dataset(train_df)
		explore_dataset(test_df)
	except Exception as e:
		logging.exception("Data load/explore failed: %s", e)

	print("Project skeleton initialized. Open this file and implement remaining functions.")


if __name__ == "__main__":
	main()
