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
# Import all required libraries for:
# - Data manipulation (pandas, numpy)
# - Visualization (matplotlib, seaborn)
# - Machine learning (scikit-learn, tensorflow/keras)
# - Data scaling and encoding tools
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------
# Define constants for the dataset file paths:
#   TRAIN_FILE = "adult.data"
#   TEST_FILE  = "adult.test"
#   NAMES_FILE = "adult.names"
# This keeps file references centralized and easy to modify.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------
# Read the CSV data into pandas DataFrames.
# Assign column names based on the 'adult.names' metadata.
# Use proper delimiters, skip initial spaces, and handle test header rows.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Explore Dataset
# ---------------------------------------------------------------
# Perform initial exploration:
# - Print dataset shapes, feature names, and types
# - View a few sample rows
# - Count and locate missing values
# - Check for duplicates or outliers
# - Summarize feature distributions and target balance
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------------------
# Identify which columns contain missing values.
# Choose imputation methods:
#   - Mode imputation for categorical features (workclass, occupation)
#   - Replace rare missing categories (e.g., native-country) with 
#     most common value or “Unknown”
# Ensure no more than two attributes are removed per project guidelines.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Encode Categorical Data
# ---------------------------------------------------------------
# Convert string-based categorical variables into numeric form
# suitable for ML algorithms:
#   - One-hot encoding (pd.get_dummies or sklearn OneHotEncoder)
#   - Label encoding (if needed)
# Align encoded columns between training and test sets.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Normalize / Standardize Continuous Features
# ---------------------------------------------------------------
# Scale numerical columns (age, hours-per-week, capital-gain, etc.)
# using StandardScaler or MinMaxScaler to ensure balanced input ranges.
# This step improves both ANN and SVM model performance.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Separate Features and Labels
# ---------------------------------------------------------------
# Split the dataset into:
#   X = input features
#   y = target variable (1 for >50K, 0 for <=50K)
# Maintain consistent shape and alignment for train/test splits.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Save Preprocessed Data (optional)
# ---------------------------------------------------------------
# Save cleaned and preprocessed DataFrames to CSV for future reuse:
#   cleaned_train.csv, cleaned_test.csv
# This helps separate data preparation from model experimentation.
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Ready for Modeling
# ---------------------------------------------------------------
# At this point, the data is ready for:
#   - Training and tuning an SVM classifier
#   - Training and tuning an ANN classifier
#   - Comparing their outputs, accuracy, and fairness
# This phase (preprocessing) sets up Phase 2: Model Development.
# ---------------------------------------------------------------
