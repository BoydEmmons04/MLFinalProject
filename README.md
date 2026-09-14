# AI/ML Final Project — ANN vs. SVM on the Adult Census Income Dataset

**Author:** Carter Ward (with Boyd Emmons)
**Course:** CS 430-1
**Date:** 12/7/25 (Final Version)

## Purpose

This is my final project for CS 430, my AI/ML course. The assignment was to build and evaluate machine learning models on a real dataset and be able to explain how they actually work under the hood, not just call a library and report a number. Boyd Emmons and I worked on this together, and I'm writing this README from my own perspective on what we built and what I took away from it.

The project trains two different classifiers — a feedforward Artificial Neural Network (ANN) and a linear Support Vector Machine (SVM) — on the UCI "Adult" Census Income dataset and compares how well each one predicts whether a person earns more or less than $50K a year. The point wasn't just to get a working model; it was to demonstrate that we understood forward passes, backpropagation, and hinge-loss gradient updates well enough to write them ourselves in NumPy instead of leaning on `scikit-learn`'s or `keras`'s built-in estimators.

## Problem and Approach

The assigned problem is a binary classification task on the classic UCI Adult dataset (`adult.data` for training, `adult.test` for testing): given 14 attributes about a person — age, workclass, education, occupation, marital status, hours worked per week, capital gain/loss, native country, etc. — predict whether their income is `<=50K` or `>50K` a year.

My approach, implemented in `FinalProject.py`, was:

1. **Load both files with the correct column names** (the raw `.data`/`.test` files have no header row), treat `"?"` as missing data, and strip the trailing periods pandas would otherwise choke on in the test file's labels.
2. **Clean the data** by imputing missing numeric fields with the column median and missing categorical fields with the column mode, so neither model ever sees a NaN.
3. **Separate features from the target** and map the income label to 0/1 (`<=50K` → 0, `>50K` → 1).
4. **One-hot encode** all nine categorical columns (workclass, education, marital-status, occupation, relationship, race, sex, native-country) using a single `OneHotEncoder` fit on the training set and reused (not refit) on the test set, so both splits land in the same feature space.
5. **Standardize everything** (numeric and one-hot columns alike) with `StandardScaler`, again fit only on training data.
6. **Hold out 20% of the training data** as a validation set (stratified on the label) so both models have something to check convergence against besides the test set.
7. **Train an ANN and a linear SVM that Boyd and I wrote ourselves from scratch using only NumPy** — no `sklearn.svm` or `keras`/`torch` doing the math for us — specifically so we could control and demonstrate every step of the forward pass, backpropagation, and hinge-loss gradient update by hand.
8. **Evaluate both models on the untouched test set** using accuracy, precision, recall, F1, and a confusion matrix, and write everything to a comparison report.

## Structure and Methodologies

**Dependencies:** `numpy`, `pandas`, `matplotlib`, and `scikit-learn` (used only for `train_test_split`, `StandardScaler`, `OneHotEncoder`, and the metric functions — never for the models themselves). Written and run in Visual Studio Code on Python 3.13.9.

**Everything lives in one script, `FinalProject.py`**, organized into clearly separated sections:

- **Data loading/cleaning helpers** — `load_dataset`, `_clean_test_labels`, `explore_dataset`, `handle_missing_values`, `split_features_labels`
- **Feature engineering helpers** — `encode_categorical`, `scale_features`
- **`ManualANN`** — a feedforward network built from raw NumPy arrays: He-initialized weight matrices, ReLU activations on the hidden layers, a sigmoid output layer, binary cross-entropy loss (with optional L2 term), and a hand-written `_backward` method that derives gradients layer by layer for mini-batch gradient descent.
- **`ManualLinearSVM`** — a linear SVM trained in the primal with a fully vectorized hinge-loss gradient: on every epoch it finds which training points violate the margin and updates `w`/`b` from a `0.5*||w||^2 + C * hinge_loss` objective.
- **Evaluation and reporting helpers** — `evaluate_models` (accuracy/precision/recall/F1/confusion matrix via `sklearn.metrics`), `save_model_comparison` (writes `model_comparison_report.txt`), `plot_ann_convergence` / `plot_svm_convergence` (matplotlib loss curves).
- **`main()`** — ties the whole pipeline together end to end, from raw CSV to trained models to saved report and plots.

**How to run it:**
```
pip install numpy pandas scikit-learn matplotlib
python FinalProject.py
```
`adult.data` and `adult.test` need to be sitting in the same folder as the script. Running it reproduces `model_comparison_report.txt`, the two preprocessed CSVs, and the two convergence PNGs. Fair warning — it takes roughly ten minutes on a normal laptop, mostly because the ANN's forward/backward passes are pure NumPy on the CPU running over many epochs and a wide one-hot-encoded feature space. That slowness is expected for a from-scratch NumPy implementation, not a sign anything is broken; `terminal_output.txt` in this repo shows what a normal run looks like.

## Process

This is roughly the order things happened as I built and ran the pipeline:

1. **Loaded the raw data.** `adult.data` came in at 32,561 rows and `adult.test` at 16,281 rows, each with the same 15 columns (14 features + `income`). I logged shapes, dtypes, and a `.head()` for both to sanity-check that the column assignment matched what `adult.names` describes.
2. **Checked for missing values.** `workclass`, `occupation`, and `native-country` were the only columns with gaps — 4,262 missing cells total in the training set and 2,203 in the test set (all marked `"?"` in the raw files). I confirmed the class balance too: about 76% `<=50K` and 24% `>50K` in both splits, so the dataset is meaningfully imbalanced.
3. **Filled the gaps** — numeric columns with their median, categorical columns with their mode — and re-checked that the missing count dropped to zero afterward.
4. **Split features from the label** and mapped `income` to 0/1.
5. **One-hot encoded the nine categorical columns**, fitting the encoder on the training data only and reusing it for the test set so column alignment couldn't drift between the two.
6. **Standardized the full feature matrix** (numeric + one-hot) with a `StandardScaler` fit on training data, cast to `float32` to keep the matrix math lighter.
7. **Carved out a stratified 80/20 train/validation split** from the training data so I had a validation loss to track during training, separate from the final test set.
8. **Trained the SVM first** — 15 epochs of full-batch hinge-loss gradient descent (`C=1.0`, learning rate `1e-3`). The training loss bounced around a fair amount (181,220 → 41,656 → 67,630 → 59,771 across the logged epochs) rather than smoothly decreasing, which told me the fixed learning rate was a bit aggressive for a full-batch update on this feature size — it's visible in `svm_convergence.png`.
9. **Trained the ANN next** — a 2-hidden-layer network (32 then 16 units, ReLU, He init) for 10 epochs with mini-batches of 256 and a learning rate of `5e-3`, tracking both train and validation binary cross-entropy each epoch. Interestingly, the loss actually crept *up* over the 10 epochs (0.6705 → 0.6963 → 0.7679), which is visible in `ann_convergence.png` and tells me the learning rate was likely too high for this architecture/epoch count — a good example of a real convergence issue rather than a synthetic one.
10. **Evaluated both models on the held-out test set** with accuracy, precision, recall, F1, and a confusion matrix.
11. **Saved the results** — `model_comparison_report.txt` for the metrics, `adult_preprocessed_train.csv` / `adult_preprocessed_test.csv` for the fully encoded/scaled feature matrices (handy for double-checking the preprocessing later), and the two convergence PNGs.

## Outcome

Final test-set results, from `model_comparison_report.txt`:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| SVM   | 0.7996   | 0.5513    | 0.8144 | 0.6575 |
| ANN   | 0.8143   | 0.6438    | 0.4789 | 0.5493 |

The ANN edged out the SVM on raw accuracy by about 1.5 points (0.8143 vs. 0.7996), but the two models actually trade off in an interesting way: the SVM catches far more of the actual `>50K` earners (recall 0.81 vs. 0.48) at the cost of more false positives, while the ANN is more conservative and precise but misses over half of the true `>50K` cases. Given the ~76/24 class imbalance in the data, accuracy alone is a little misleading — a model that mostly predicts the majority class can look "good" on accuracy while doing poorly on the minority class, which is exactly why the report also tracks precision/recall/F1 and confusion matrices instead of just accuracy. Looking at the confusion matrices, the SVM overpredicts `>50K` (712 false negatives vs. 2,549 false positives) while the ANN underpredicts it (2,004 false negatives vs. only 1,019 false positives) — so which model is "better" genuinely depends on whether false negatives or false positives matter more for the use case.

Working through this project, I came away with a much better hands-on grasp of a few things I'd only seen in lecture before:

- **What backpropagation is actually doing**, step by step — deriving the gradient of binary cross-entropy through a sigmoid output and ReLU hidden layers, instead of just calling `.backward()`.
- **How an SVM's hinge-loss gradient update works** in the primal, and what "support vectors" (the margin-violating points that drive the gradient) look like in code.
- **Why preprocessing choices matter** — fitting the encoder/scaler on training data only, handling missing values sensibly, and keeping train/test feature spaces aligned are the kind of details that silently break a model if you get them wrong.
- **How to read a convergence plot critically.** Both plots in this repo show real, slightly messy training behavior (the SVM's noisy full-batch updates, the ANN's rising loss from too-high a learning rate) rather than a textbook-smooth curve, and being able to diagnose "the learning rate is probably too high" from a loss curve going the wrong direction is a skill I only really solidified by watching it happen on my own model.
- **Why accuracy alone isn't enough** on an imbalanced dataset, and how precision/recall/F1 tell a more honest story about what a classifier is actually getting right.

Overall, this project demonstrates that I can take a real, messy dataset through a full ML pipeline — cleaning, encoding, scaling, training, and evaluating — and that I understand the math inside the models well enough to implement and debug it myself rather than treating it as a black box.
