# -------------------------------------------------------------------
# Authors: Carter Ward, Boyd Emmons
# Course : CS 430 - Section 1
# Date   : 12/5/25
#
# Project Summary:
#   This program builds two models, an Artificial Neural Network (ANN)
#   and a Support Vector Machine (SVM), to predict whether a person
#   earns more than $50K a year using the Adult Census dataset.
#
#   We wrote the ANN and SVM ourselves using NumPy so we could see
#   exactly how the learning process works, instead of relying on
#   machine learning libraries that hide most of the details. This let
#   us control the steps for forward passes, backpropagation, and the
#   SVM's hinge-loss updates.
#
#   The program loads and cleans the data, encodes and scales the
#   features, trains both models, and then evaluates them using common
#   metrics. It also saves a comparison report that summarizes how the
#   two models performed.
#
#   In addition, it generates convergence plots showing how the ANN and
#   SVM training and validation losses change over epochs.
# -------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a Census Income dataset file and convert it to a DataFrame.

    - Assigns the known column names
    - Handles '?' markers as missing values
    - Strips unnecessary whitespace
    - Applies test-file specific cleanup rules
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

    # The provided test file has a header line to skip
    if is_test:
        read_kwargs["skiprows"] = 1

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
    """Log basic info about the dataset just for sanity checking."""
    logger = logging.getLogger("EDA")

    logger.info("Shape: %s", (df.shape,))
    logger.info("Dtypes: %s", df.dtypes)

    with pd.option_context("display.max_rows", 5, "display.max_columns", 20):
        logger.info("Head: %s", df.head())

    na_counts = df.isna().sum().sort_values(ascending=False)
    logger.info("Missing values by column: %s", na_counts)

    if "income" in df.columns:
        class_counts = df["income"].value_counts(dropna=False)
        class_ratio = (class_counts / class_counts.sum()).round(4)
        logger.info("Class distribution: %s", class_counts.to_string())
        logger.info("Class proportions: %s", class_ratio.to_string())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    logger.info("Numeric fields (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical fields (%d): %s", len(cat_cols), cat_cols)

    if num_cols:
        with pd.option_context("display.max_rows", 100, "display.max_columns", 20):
            logger.info("Numeric summary: %s", df[num_cols].describe().T)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing entries.
    - Numeric columns: median
    - Categorical columns: mode
    """
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
    """Separate predictors from the target and convert labels to 0/1."""
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
    """
    One-hot encode categorical columns.
    Reuse the same encoder object between train and test.
    """
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
    """
    Standardize features using z-score.
    Always output float32 to keep computations smaller/faster.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), scaler


def save_preprocessed(df: pd.DataFrame, out_path: Path) -> None:
    """Save processed data to CSV so we can inspect later if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.getLogger("IO").info("Saved preprocessed data: %s", out_path)


# -------------------------------------------------------------------
# Model comparison output function (SVM vs ANN)
# -------------------------------------------------------------------
def save_model_comparison(metrics: Dict[str, dict], out_path: Path) -> None:
    """
    Generate a clean comparison report (ANN vs SVM) and save it to a file.
    Includes accuracy, precision, recall, F1, and confusion matrices.
    """
    lines = []
    lines.append("====================================================")
    lines.append("        MODEL PERFORMANCE COMPARISON REPORT         ")
    lines.append("====================================================\n")

    for model_name, m in metrics.items():
        lines.append(f"Model: {model_name}")
        lines.append("-" * (7 + len(model_name)))
        lines.append(f"Accuracy         : {m['accuracy']:.4f}")
        lines.append(f"Precision        : {m['precision']:.4f}")
        lines.append(f"Recall           : {m['recall']:.4f}")
        lines.append(f"F1 Score         : {m['f1']:.4f}")
        lines.append("Confusion Matrix :")
        lines.append(str(m["confusion_matrix"]))
        lines.append("")

    lines.append("====================================================")
    lines.append("                     SUMMARY                        ")
    lines.append("====================================================\n")

    if "SVM" in metrics and "ANN" in metrics:
        svm_acc = metrics["SVM"]["accuracy"]
        ann_acc = metrics["ANN"]["accuracy"]

        if ann_acc > svm_acc:
            lines.append(f"ANN outperformed SVM by {ann_acc - svm_acc:.4f} accuracy points.")
        elif svm_acc > ann_acc:
            lines.append(f"SVM outperformed ANN by {svm_acc - ann_acc:.4f} accuracy points.")
        else:
            lines.append("ANN and SVM achieved identical accuracy.")

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------
# Manual ANN implementation (with mini-batch training)
# -------------------------------------------------------------------
class ManualANN:
    """
    Very simple feedforward neural network for binary classification.
    Uses ReLU for hidden layers and sigmoid for the output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(32, 16),
        learning_rate: float = 5e-3,
        l2_reg: float = 0.0,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        rng = np.random.RandomState(seed)
        layer_sizes = [input_dim] + self.hidden_dims + [1]

        self.W = []
        self.b = []

        # Random weight initialization (He init for ReLU)
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = rng.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / in_size)
            b = np.zeros(out_size, dtype=np.float32)
            self.W.append(W)
            self.b.append(b)

        # For convergence plots
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def _forward(self, X: np.ndarray):
        """
        Forward pass.
        Returns final activations and a list of caches (A_prev, Z) for backprop.
        """
        A = X
        caches = []
        L = len(self.W)

        for l in range(L):
            Z = A @ self.W[l] + self.b[l]
            if l < L - 1:
                A_next = self._relu(Z)
            else:
                A_next = self._sigmoid(Z)
            caches.append((A, Z))
            A = A_next

        return A, caches

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss + optional L2 regularization."""
        m = y_true.shape[0]
        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )

        if self.l2_reg > 0.0:
            l2_sum = sum(np.sum(W * W) for W in self.W)
            loss += (self.l2_reg / (2.0 * m)) * l2_sum

        return float(loss)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray, caches):
        """
        Backpropagation to compute gradients w.r.t. weights and biases.
        """
        m = y_true.shape[0]
        grads_W = [np.zeros_like(W) for W in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]

        y = y_true.reshape(-1, 1)
        L = len(self.W)

        # Output layer
        A_prev, Z_L = caches[-1]
        dZ = y_pred - y  # derivative of CE loss with sigmoid
        grads_W[L - 1] = (A_prev.T @ dZ) / m + self.l2_reg * self.W[L - 1] / m
        grads_b[L - 1] = dZ.sum(axis=0) / m
        dA_prev = dZ @ self.W[L - 1].T

        # Hidden layers (ReLU)
        for l in reversed(range(L - 1)):
            A_prev, Z = caches[l]
            dZ = dA_prev * (Z > 0.0).astype(np.float32)
            grads_W[l] = (A_prev.T @ dZ) / m + self.l2_reg * self.W[l] / m
            grads_b[l] = dZ.sum(axis=0) / m
            dA_prev = dZ @ self.W[l].T

        return grads_W, grads_b

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Train the network with mini-batch gradient descent.
        Also records training and validation loss per epoch.
        """
        logger = logging.getLogger("ANN")
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.float32)
        m = X.shape[0]

        self.train_losses.clear()
        self.val_losses.clear()

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32)

        for epoch in range(1, epochs + 1):
            # Shuffle the training data each epoch
            idx = np.random.permutation(m)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # Go through mini-batches
            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if X_batch.shape[0] == 0:
                    continue

                y_pred, caches = self._forward(X_batch)
                grads_W, grads_b = self._backward(y_batch, y_pred, caches)

                # Gradient descent step
                for l in range(len(self.W)):
                    self.W[l] -= self.learning_rate * grads_W[l]
                    self.b[l] -= self.learning_rate * grads_b[l]

            # Compute full-epoch training loss
            y_pred_full, _ = self._forward(X)
            train_loss = self._compute_loss(y, y_pred_full)
            self.train_losses.append(train_loss)

            # Compute validation loss if provided
            if has_val:
                y_val_pred, _ = self._forward(X_val_arr)
                val_loss = self._compute_loss(y_val_arr, y_val_pred)
                self.val_losses.append(val_loss)
                msg = (
                    f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} "
                    f"| Val loss: {val_loss:.4f}"
                )
            else:
                self.val_losses.append(np.nan)
                msg = f"Epoch {epoch:03d} | Train loss: {train_loss:.4f}"

            # Only log every few epochs to keep things light
            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                logger.info(msg)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for the positive class."""
        X = np.asarray(X, dtype=np.float32)
        y_pred, _ = self._forward(X)
        return y_pred.reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 0/1 predictions based on 0.5 threshold."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


def build_ann_model(input_shape: int) -> ManualANN:
    """Create the manual ANN with smaller/faster architecture."""
    model = ManualANN(
        input_dim=input_shape,
        hidden_dims=(32, 16),
        learning_rate=5e-3,
        l2_reg=0.0,
        batch_size=256,
    )
    return model


# -------------------------------------------------------------------
# Manual linear SVM implementation (vectorized)
# -------------------------------------------------------------------
class ManualLinearSVM:
    """
    Linear SVM trained in the primal using hinge loss and
    simple gradient descent with fully vectorized updates.
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 1e-3, n_epochs: int = 15):
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

        # For convergence plots
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _compute_hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute regularized hinge loss:
        0.5 * ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b))).
        y is expected to be 0/1.
        """
        if self.w is None:
            raise ValueError("Model not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        y_signed = np.where(y == 1, 1.0, -1.0).astype(np.float32)

        scores = X @ self.w + self.b
        margins = 1.0 - y_signed * scores
        hinge = np.maximum(0.0, margins).sum()
        reg = 0.5 * np.sum(self.w * self.w)
        loss = reg + self.C * hinge
        return float(loss)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train linear SVM with vectorized hinge loss gradient.
        Labels y are 0/1; they are converted to -1/+1 here.
        Also records training and validation hinge loss per epoch.
        """
        logger = logging.getLogger("SVM")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        y_signed = np.where(y == 1, 1.0, -1.0).astype(np.float32)

        m, n = X.shape
        self.w = np.zeros(n, dtype=np.float32)
        self.b = 0.0

        self.train_losses.clear()
        self.val_losses.clear()

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.int32)

        for epoch in range(1, self.n_epochs + 1):
            # Compute margins for all points at once
            scores = X @ self.w + self.b
            margins = y_signed * scores

            # Points that violate the margin
            misclassified = margins < 1.0

            # If everything is correctly classified, gradient only has w term
            if np.any(misclassified):
                X_mis = X[misclassified]
                y_mis = y_signed[misclassified]

                dw = self.w - self.C * (X_mis * y_mis[:, None]).sum(axis=0)
                db = -self.C * y_mis.sum()
            else:
                dw = self.w
                db = 0.0

            # Gradient descent step
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Track hinge loss
            train_loss = self._compute_hinge_loss(X, y)
            self.train_losses.append(train_loss)

            if has_val:
                val_loss = self._compute_hinge_loss(X_val_arr, y_val_arr)
                self.val_losses.append(val_loss)
                msg = (
                    f"Epoch {epoch:03d} | SVM train loss: {train_loss:.4f} "
                    f"| Val loss: {val_loss:.4f}"
                )
            else:
                self.val_losses.append(np.nan)
                msg = f"Epoch {epoch:03d} | SVM train loss: {train_loss:.4f}"

            # Light logging just to see training is doing something
            if epoch == 1 or epoch % 5 == 0 or epoch == self.n_epochs:
                logger.info(msg)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.w is None:
            raise ValueError("Model not fitted yet.")
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)


def build_svm_model() -> ManualLinearSVM:
    """Create a linear SVM with modest number of epochs for speed."""
    svm = ManualLinearSVM(
        C=1.0,
        learning_rate=1e-3,
        n_epochs=15,
    )
    return svm


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_models(models: Dict[str, Any], X_test: Any, y_test: Any) -> Dict[str, dict]:
    """Evaluate each model and compute standard classification metrics."""
    results = {}
    logger = logging.getLogger("Eval")

    y_true = np.asarray(y_test, dtype=int)

    for name, model in models.items():
        if model is None:
            logger.warning("Model '%s' is missing; skipping.", name)
            continue

        if not hasattr(model, "predict"):
            raise ValueError(f"Model '{name}' lacks a predict() method.")

        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred).reshape(-1)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        logger.info(
            "[%s] Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            name,
            acc,
            prec,
            rec,
            f1,
        )
        logger.info("[%s] Confusion matrix: %s", name, cm)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
        }

    return results


# -------------------------------------------------------------------
# Convergence plotting helpers
# -------------------------------------------------------------------
def plot_ann_convergence(ann_model: ManualANN, out_path: Path) -> None:
    """Plot ANN training vs validation loss in a single graph."""
    epochs = range(1, len(ann_model.train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, ann_model.train_losses, label="Training Loss")
    if ann_model.val_losses and not np.all(np.isnan(ann_model.val_losses)):
        plt.plot(epochs, ann_model.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ANN Convergence (Loss vs Epoch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_svm_convergence(svm_model: ManualLinearSVM, out_path: Path) -> None:
    """Plot SVM training vs validation hinge loss in a single graph."""
    epochs = range(1, len(svm_model.train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, svm_model.train_losses, label="Training Loss")
    if svm_model.val_losses and not np.all(np.isnan(svm_model.val_losses)):
        plt.plot(epochs, svm_model.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SVM Convergence (Hinge Loss vs Epoch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------
# Main execution pipeline
# -------------------------------------------------------------------
def main() -> None:
    """Run the full processing and modeling sequence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
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

    # --- 2. Preliminary inspection (logs only; no printing to console) ---
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

    # Validation split for both ANN and SVM
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
    logger.info("Training manual SVM...")
    svm_model = build_svm_model()
    svm_model.fit(
        X_split_train,
        y_split_train.values,
        X_val=X_split_val,
        y_val=y_split_val.values,
    )

    logger.info("Training manual ANN...")
    ann_model = build_ann_model(X_train_scaled.shape[1])
    ann_model.fit(
        X_split_train,
        y_split_train.values,
        X_val=X_split_val,
        y_val=y_split_val.values,
        epochs=10,
        verbose=True,
    )

    # --- 8. Evaluation ---
    logger.info("Evaluating models...")
    models = {"SVM": svm_model, "ANN": ann_model}
    metrics = evaluate_models(models, X_test_scaled, y_test.values)
    logger.info("Evaluation results: %s", metrics)

    # --- 9A. Save comparison report (SVM vs ANN) ---
    report_path = ROOT / "model_comparison_report.txt"
    save_model_comparison(metrics, report_path)
    logger.info("Model comparison report saved to: %s", report_path)

    # --- 9B. Save processed data (optional, but handy for debugging) ---
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

    # --- 10. Convergence plots ---
    plot_ann_convergence(ann_model, ROOT / "ann_convergence.png")
    plot_svm_convergence(svm_model, ROOT / "svm_convergence.png")
    logger.info("Convergence plots saved as 'ann_convergence.png' and 'svm_convergence.png'.")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
