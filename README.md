Project Overview

This project builds an Artificial Neural Network (ANN) and a Support Vector Machine (SVM) to compare their performance on a binary classification task. The models predict whether an individual’s income exceeds $50K per year using the Adult Census Income dataset. Both models use the same cleaned, encoded, and scaled data so their results can be compared directly.

Development Environment

• Editor Used: Visual Studio Code
• Python Version: 3.13.9 (Microsoft Store version – Stable)
• Libraries Used: NumPy, Pandas, Scikit-Learn (preprocessing and metrics only)

The models themselves do not rely on advanced machine learning libraries. The neural network and SVM are implemented through NumPy operations, which allows complete control over how the models learn.

Runtime Information

The program may take one to two minutes to run. This is expected for several reasons:

The ANN performs forward and backward passes using NumPy, which is not as fast as optimized GPU frameworks.

The SVM trains with gradient updates over multiple epochs, which involves repeated matrix operations across thousands of features created during one-hot encoding.

Preprocessing expands the dataset significantly, and scaling plus encoding adds additional computation before training begins.

Even though the training loops are optimized, the dataset size and number of features naturally lead to a noticeable runtime.

How to Run the Program

Make sure Python 3.13.9 (or a compatible Python 3 version) is installed.

Install required packages:

pip install numpy pandas scikit-learn


Place adult.data and adult.test in the project directory.

Run the program from VS Code or from the terminal:

python main.py


The program will:
• Load and clean the data
• One-hot encode and scale all features
• Train both the ANN and the SVM
• Evaluate each model
• Save a comparison report and the preprocessed datasets

Output Files

The program produces several files:
• adult_preprocessed_train.csv – Full transformed training set
• adult_preprocessed_test.csv – Full transformed test set
• model_comparison_report.txt – Accuracy, precision, recall, F1, and confusion matrix results for both models
