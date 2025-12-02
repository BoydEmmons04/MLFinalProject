# -------------------------------------------------------------------
# Authors: Carter Ward, Boyd Emmons
# Course : CS 430 - Section 1
# Date   : 12/2/25
#
# Project Summary:
#   This program develops two predictive models—a neural network and an
#   SVM—to estimate whether an individual's income surpasses $50K per
#   year using the Adult Census dataset. The workflow includes data
#   loading, preparation, feature engineering, model training, and
#   evaluation. 
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
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

# Deep learning tools (available only if the environment supports them)
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None


# -------------------------------------------------------------------
# File paths and constants
# -------------------------------------------------------------------
ROOT = Path(__file__).parent
TRAIN_FILE: Path = ROOT / "adult.data"
TEST_FILE: Path = ROOT / "adult.test"
NAMES_FILE: Path = ROOT / "adult.names"

# Expected column names for the Adult dataset
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


# -------------------------------------------------------------------
# Helper routines for data loading and cleanup
# -------------------------------------------------------------------
def _is_test_file(p: Path) -> bool:
    """Check whether the file corresponds to the testing split."""
    return p.name.lower().endswith("adult.test")


def _clean_test_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing periods from the income labels in the test data."""
    if "income" in df.columns and pd.api.types.is_object_dtype(df["income"]):
        df["income"] = df["income"].str.strip().str.replace(r"\.$", "", regex=True)
    return df


# -------------------------------------------------------------------
# Data ingestion and preparation functions
# -------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Load a Census Income dataset file and convert it to a DataFrame.

    - Assigns the known column names
    - Handles '?' markers as missing values
    - Strips unnecessary whitespace
    - Applies test-file specific cleanup rules

    Raises:
        FileNotFoundError: if the file cannot be located.
        ValueError: if the column count does not match expectations.
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
        read_kwargs["skiprows"] = 1  # test file has a header line

    df = pd.read_csv(path, **read_kwargs)
    if is_test:
        df = _clean_test_labels(df)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected column count in {path.name}: "
            f"got {df.shape[1]}, expected {len(COLUMN_NAMES)}"
        )
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Log a basic exploratory summary of the dataset structure."""
    logger = logging.getLogger("EDA")

    logger.info("Shape: %s", (df.shape,))
    logger.info("Dtypes:\n%s", df.dtypes)

    with pd.option_context("display.max_rows", 5, "display.max_columns", 20):
        logger.info("Head:\n%s", df.head())

    na_counts = df.isna().sum().sort_values(ascending=False)
    logger.info("Missing values by column:\n%s", na_counts)

    if "income" in df.columns:
        class_counts = df["income"].value_counts(dropna=False)
        class_ratio = (class_counts / class_counts.sum()).round(4)
        logger.info("Class distribution:\n%s", class_counts.to_string())
        logger.info("Class proportions:\n%s", class_ratio.to_string())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    logger.info("Numeric fields (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical fields (%d): %s", len(cat_cols), cat_cols)

    if num_cols:
        with pd.option_context("display.max_rows", 100, "display.max_columns", 20):
            logger.info("Numeric summary:\n%s", df[num_cols].describe().T)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing entries using a median strategy for numeric fields
    and a mode strategy for categorical fields."""
    logger = logging.getLogger("MissingValues")
    df = df.copy()

    before = int(df.isna().sum().sum())
    logger.info("Missing values before processing: %d", before)

    target_col = "income"
    feature_cols = [c for c in df.columns if c != target_col]

    num_cols = df[feature_cols].select_dtypes(exclude="object").columns
    cat_cols = df[feature_cols].select_dtypes(include="object").columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series.iloc[0])

    after = int(df.isna().sum().sum())
    logger.info("Missing values after processing: %d", after)

    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the target and convert labels to binary."""
    if "income" not in df.columns:
        raise ValueError("Expected 'income' column to be present.")

    y_text = df["income"].astype(str).str.strip()
    mapping = {"<=50K": 0, ">50K": 1}

    y = y_text.map(mapping)
    if y.isna().any():
        missing_vals = sorted(y_text[y.isna()].unique())
        raise ValueError(f"Unexpected label values encountered: {missing_vals}")

    X = df.drop(columns=["income"])
    return X, y.astype(int)


def encode_categorical(
    df: pd.DataFrame, encoder: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """Convert categorical fields into a one-hot encoded representation."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()

    if encoder is None:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_array = (
            encoder.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
        )
    else:
        cat_array = (
            encoder.transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
        )

    if cat_cols:
        cat_df = pd.DataFrame(
            cat_array,
            columns=encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )
    else:
        cat_df = pd.DataFrame(index=df.index)

    num_df = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

    encoded_df = pd.concat([num_df, cat_df], axis=1)
    return encoded_df, encoder


def scale_features(
    X: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize numerical features using the z-score transformation."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), scaler


def save_preprocessed(df: pd.DataFrame, out_path: Path) -> None:
    """Write a processed DataFrame to a CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.getLogger("IO").info("Saved preprocessed data: %s", out_path)


# -------------------------------------------------------------------
# Model building routines
# -------------------------------------------------------------------
def build_ann_model(input_shape: int) -> Any:
    """Construct a feedforward neural network for binary classification."""
    if keras is None:
        logging.warning("Keras not available; ANN model creation skipped.")
        return None

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
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )
    return model


def build_svm_model() -> SVC:
    """Create an RBF-kernel SVM tuned for imbalanced classification."""
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )
    return svm


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_models(models: Dict[str, Any], X_test: Any, y_test: Any) -> Dict[str, dict]:
    """Evaluate each model and compute standard classification metrics."""
    results = {}
    logger = logging.getLogger("Eval")

    for name, model in models.items():
        if model is None:
            logger.warning("Model '%s' is missing; skipping.", name)
            continue

        if not hasattr(model, "predict"):
            raise ValueError(f"Model '{name}' lacks a predict() method.")

        y_pred = model.predict(X_test)

        if not isinstance(model, SVC):
            y_pred = np.asarray(y_pred).reshape(-1)
            y_pred = (y_pred >= 0.5).astype(int)
        else:
            y_pred = np.asarray(y_pred).reshape(-1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(
            "[%s] Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            name, acc, prec, rec, f1,
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


# -------------------------------------------------------------------
# Main execution pipeline
# -------------------------------------------------------------------
def main() -> None:
    """Run the full processing and modeling sequence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info("Pipeline started.")

    # --- 1. Load datasets ---
    try:
        train_df = load_dataset(TRAIN_FILE)
        test_df = load_dataset(TEST_FILE)
    except Exception as e:
        logger.exception("Failed to load data: %s", e)
        return

    # --- 2. Preliminary inspection ---
    logger.info("Inspecting training data...")
    explore_dataset(train_df)
    logger.info("Inspecting test data...")
    explore_dataset(test_df)

    # --- 3. Missing value resolution ---
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    # --- 4. Feature/label separation ---
    X_train_df, y_train = split_features_labels(train_df)
    X_test_df, y_test = split_features_labels(test_df)

    # --- 5. Categorical encoding ---
    X_train_encoded, encoder = encode_categorical(X_train_df, encoder=None)
    X_test_encoded, _ = encode_categorical(X_test_df, encoder=encoder)

    # --- 6. Scaling ---
    X_train_scaled, scaler = scale_features(X_train_encoded, scaler=None)
    X_test_scaled, _ = scale_features(X_test_encoded, scaler=scaler)

    # Create validation subset for ANN
    (
        X_split_train,
        X_split_val,
        y_split_train,
        y_split_val,
    ) = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # --- 7. Train models ---
    logger.info("Training SVM...")
    svm_model = build_svm_model()
    svm_model.fit(X_train_scaled, y_train)

    if keras is not None:
        logger.info("Training ANN...")
        ann_model = build_ann_model(X_train_scaled.shape[1])
        history = ann_model.fit(
            X_split_train,
            y_split_train.values,
            validation_data=(X_split_val, y_split_val.values),
            epochs=20,
            batch_size=256,
            verbose=0,
        )
        logger.info(
            "ANN training completed. Final training accuracy: %.4f",
            float(history.history["accuracy"][-1]),
        )
    else:
        ann_model = None
        logger.warning("ANN model skipped (TensorFlow/Keras unavailable).")

    # --- 8. Evaluation ---
    logger.info("Evaluating models...")
    models = {"SVM": svm_model, "ANN": ann_model}
    metrics = evaluate_models(models, X_test_scaled, y_test.values)
    logger.info("Evaluation results: %s", metrics)

    # --- 9. Save processed data ---
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

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
