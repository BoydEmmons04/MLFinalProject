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
# Imports
# ---------------------------------------------------------------
import logging
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Optional deep learning import (may not be installed in all envs)
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:  # pragma: no cover - optional dependency
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
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


# ---------------------------------------------------------------
# Internal helpers for dataset loading/cleaning
# ---------------------------------------------------------------
def _is_test_file(p: Path) -> bool:
    return p.name.lower().endswith("adult.test")


def _clean_test_labels(df: pd.DataFrame) -> pd.DataFrame:
    """adult.test labels include a trailing '.' (e.g., '>50K.'); strip it."""
    if "income" in df.columns and pd.api.types.is_object_dtype(df["income"]):
        df["income"] = df["income"].str.strip().str.replace(r"\.$", "", regex=True)
    return df


# ---------------------------------------------------------------
# Pipeline function implementations
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
        na_values=["?"],
        skipinitialspace=True,
    )
    if is_test:
        # adult.test has a first line that's not data
        read_kwargs["skiprows"] = 1

    df = pd.read_csv(path, **read_kwargs)
    if is_test:
        df = _clean_test_labels(df)

    # Sanity check: expect exactly 15 columns
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected column count in {path.name}: "
            f"got {df.shape[1]}, expected {len(COLUMN_NAMES)}"
        )
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Lightweight EDA: shape, dtypes, missingness, class balance, and numeric summary."""
    logger = logging.getLogger("EDA")

    logger.info("Shape: %s", (df.shape,))
    logger.info("Dtypes:\n%s", df.dtypes)

    with pd.option_context("display.max_rows", 5, "display.max_columns", 20):
        logger.info("Head:\n%s", df.head())

    # Missing values overview
    na_counts = df.isna().sum().sort_values(ascending=False)
    logger.info("Missing values per column (desc):\n%s", na_counts)

    # Target distribution (if present)
    if "income" in df.columns:
        class_counts = df["income"].value_counts(dropna=False)
        class_ratio = (class_counts / class_counts.sum()).round(4)
        logger.info("Target distribution (counts):\n%s", class_counts.to_string())
        logger.info("Target distribution (proportions):\n%s", class_ratio.to_string())

    # Basic type split
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    logger.info("Numeric columns (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical columns (%d): %s", len(cat_cols), cat_cols)

    # Descriptive statistics for numeric features
    if num_cols:
        with pd.option_context("display.max_rows", 100, "display.max_columns", 20):
            logger.info("Numeric summary:\n%s", df[num_cols].describe().T)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and impute missing values.

    Strategy (satisfies 'do not remove more than two attributes'):
      - Numeric features: impute with median
      - Categorical features: impute with mode (most frequent value)
      - Target 'income' left untouched (it has no official missing values)
    """
    logger = logging.getLogger("MissingValues")
    df = df.copy()

    total_missing_before = int(df.isna().sum().sum())
    logger.info("Total missing values BEFORE imputation: %d", total_missing_before)

    target_col = "income"
    feature_cols = [c for c in df.columns if c != target_col]

    num_cols = df[feature_cols].select_dtypes(exclude="object").columns
    cat_cols = df[feature_cols].select_dtypes(include="object").columns

    # Numeric: median
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        logger.debug("Imputed numeric column '%s' with median %.3f", col, median_val)

    # Categorical: mode
    for col in cat_cols:
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            mode_val = mode_series.iloc[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed categorical column '%s' with mode '%s'", col, mode_val)

    total_missing_after = int(df.isna().sum().sum())
    logger.info("Total missing values AFTER imputation: %d", total_missing_after)

    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and labels (y) from preprocessed df.

    - Drops 'income' from features
    - Maps labels to binary (<=50K -> 0, >50K -> 1)
    """
    if "income" not in df.columns:
        raise ValueError("Expected 'income' column to be present.")

    y_raw = df["income"].astype(str).str.strip()
    label_map = {"<=50K": 0, ">50K": 1}

    y = y_raw.map(label_map)
    if y.isna().any():
        bad_vals = sorted(y_raw[y.isna()].unique())
        raise ValueError(f"Unexpected label values encountered: {bad_vals}")

    X = df.drop(columns=["income"])
    return X, y.astype(int)


def encode_categorical(
    df: pd.DataFrame, encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical columns using one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-only DataFrame (no target column).
    encoder : OneHotEncoder or None
        If None, fit a new encoder. Otherwise, transform using the
        passed encoder (for test/validation sets).

    Returns
    -------
    encoded_df : pd.DataFrame
        DataFrame with numeric columns untouched and categorical
        columns replaced by their one-hot expansions.
    encoder : OneHotEncoder
        Fitted encoder (new or the one passed in).
    """
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    if encoder is None:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )
        cat_array = encoder.fit_transform(df[cat_cols]) if cat_cols else np.empty(
            (len(df), 0)
        )
    else:
        cat_array = encoder.transform(df[cat_cols]) if cat_cols else np.empty(
            (len(df), 0)
        )

    if cat_cols:
        cat_feature_names = encoder.get_feature_names_out(cat_cols)
        cat_df = pd.DataFrame(cat_array, columns=cat_feature_names, index=df.index)
    else:
        cat_df = pd.DataFrame(index=df.index)

    num_df = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

    encoded_df = pd.concat([num_df, cat_df], axis=1)
    return encoded_df, encoder


def scale_features(
    X: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """Scale numeric features and return (X_scaled, scaler).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric) after encoding.
    scaler : StandardScaler or None
        If None, fit a new scaler. Otherwise, transform using the
        passed scaler (for test/validation sets).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), scaler


def save_preprocessed(df: pd.DataFrame, out_path: Path) -> None:
    """Save cleaned/preprocessed DataFrame to CSV for reuse."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.getLogger("IO").info("Saved preprocessed data to %s", out_path)


def build_ann_model(input_shape: int) -> Any:
    """Build and compile a binary ANN for the Adult income task.

    Architecture:
      Input -> Dense(64, ReLU) -> Dropout(0.2)
            -> Dense(32, ReLU) -> Dropout(0.2)
            -> Dense(1, Sigmoid)

    Returns:
      keras.Model (compiled), or None if Keras is unavailable.
    """
    if keras is None:
        logging.warning("Keras not available in this environment; ANN will be skipped.")
        return None

    # Optional, reproducible behavior if TF is present
    try:
        if tf is not None:
            tf.random.set_seed(42)
    except Exception:
        pass

    inputs = keras.Input(shape=(input_shape,), name="features")
    x = keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="income")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ann_income_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_svm_model() -> SVC:
    """Return an untrained SVM classifier instance.

    Here we use an RBF-kernel SVC with class_weight='balanced' to
    handle slight class imbalance in the Adult dataset.
    """
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )
    return svm


def evaluate_models(models: Dict[str, Any], X_test: Any, y_test: Any) -> Dict[str, dict]:
    """Evaluate trained models on test data and return a metrics dict.

    For each model we compute:
      - Accuracy
      - Precision
      - Recall
      - F1 score
      - Confusion matrix
    """
    results: Dict[str, dict] = {}
    logger = logging.getLogger("Eval")

    for name, model in models.items():
        if model is None:
            logger.warning("Model '%s' is None; skipping.", name)
            continue

        # Get predictions
        if hasattr(model, "predict"):
            y_pred = model.predict(X_test)

            # Keras models output probabilities in shape (N, 1)
            if not isinstance(model, SVC):
                y_pred = np.asarray(y_pred).reshape(-1)
                y_pred = (y_pred >= 0.5).astype(int)
            else:
                y_pred = np.asarray(y_pred).reshape(-1)
        else:
            raise ValueError(f"Model '{name}' does not implement predict().")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(
            "[%s] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            name,
            acc,
            prec,
            rec,
            f1,
        )
        logger.info("[%s] Confusion matrix:\n%s", name, cm)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
        }

    return results


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def main() -> None:
    """Main entry for the pipeline.

    Steps:
      1. Load train and test datasets
      2. Explore basic properties (logged)
      3. Handle missing values
      4. Split features/labels
      5. Encode categorical variables
      6. Scale features
      7. Train SVM and ANN
      8. Evaluate and print metrics
      9. Save preprocessed datasets
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info("Starting Census Income ML pipeline.")

    # -----------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------
    try:
        train_df = load_dataset(TRAIN_FILE)
        test_df = load_dataset(TEST_FILE)
    except Exception as e:
        logger.exception("Data loading failed: %s", e)
        return

    # -----------------------------------------------------------
    # 2. Simple exploration (logged)
    # -----------------------------------------------------------
    logger.info("Exploring training data...")
    explore_dataset(train_df)
    logger.info("Exploring test data...")
    explore_dataset(test_df)

    # -----------------------------------------------------------
    # 3. Handle missing values
    # -----------------------------------------------------------
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # -----------------------------------------------------------
    # 4. Split features and labels
    # -----------------------------------------------------------
    X_train_df, y_train = split_features_labels(train_df)
    X_test_df, y_test = split_features_labels(test_df)

    # -----------------------------------------------------------
    # 5. Encode categorical features
    #    (fit on train, reuse encoder on test)
    # -----------------------------------------------------------
    X_train_encoded, ohe = encode_categorical(X_train_df, encoder=None)
    X_test_encoded, _ = encode_categorical(X_test_df, encoder=ohe)

    # -----------------------------------------------------------
    # 6. Scale features
    #    (fit scaler on train, reuse on test)
    # -----------------------------------------------------------
    X_train_scaled, scaler = scale_features(X_train_encoded, scaler=None)
    X_test_scaled, _ = scale_features(X_test_encoded, scaler=scaler)

    # Optionally create a validation split for ANN
    (
        X_train_split,
        X_val_split,
        y_train_split,
        y_val_split,
    ) = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # -----------------------------------------------------------
    # 7. Train models
    # -----------------------------------------------------------
    # 7a. SVM
    logger.info("Training SVM model...")
    svm_model = build_svm_model()
    svm_model.fit(X_train_scaled, y_train)

    # 7b. ANN
    if keras is not None:
        logger.info("Training ANN model...")
        ann_model = build_ann_model(X_train_scaled.shape[1])
        history = ann_model.fit(
            X_train_split,
            y_train_split.values,
            validation_data=(X_val_split, y_val_split.values),
            epochs=20,
            batch_size=256,
            verbose=0,
        )
        logger.info(
            "ANN training complete. Final training accuracy: %.4f",
            float(history.history["accuracy"][-1]),
        )
    else:
        ann_model = None
        logger.warning("Keras not available; ANN training skipped.")

    # -----------------------------------------------------------
    # 8. Evaluate models on held-out test set
    # -----------------------------------------------------------
    logger.info("Evaluating models on test set...")
    models = {"SVM": svm_model, "ANN": ann_model}
    metrics_dict = evaluate_models(models, X_test_scaled, y_test.values)

    logger.info("Evaluation summary: %s", metrics_dict)

    # -----------------------------------------------------------
    # 9. Save preprocessed datasets (for reproducibility/reporting)
    # -----------------------------------------------------------
    train_preprocessed = pd.DataFrame(
        np.column_stack([X_train_scaled, y_train.values]),
        columns=list(X_train_encoded.columns) + ["income_binary"],
    )
    test_preprocessed = pd.DataFrame(
        np.column_stack([X_test_scaled, y_test.values]),
        columns=list(X_test_encoded.columns) + ["income_binary"],
    )

    save_preprocessed(train_preprocessed, ROOT / "adult_preprocessed_train.csv")
    save_preprocessed(test_preprocessed, ROOT / "adult_preprocessed_test.csv")

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
