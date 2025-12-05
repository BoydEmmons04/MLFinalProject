**Project Overview**

- This project develops an Artificial Neural Network (ANN) and a Support Vector Machine (SVM) to compare their performance on a binary classification task. 
- The models predict whether an individual’s income exceeds $50,000 per year using the Adult Census Income dataset. 
- Both models use the same cleaned, encoded, and scaled feature set to ensure a fair comparison.


**Develop Environment**

- Editor: Visual Studio Code
- Python Version: 3.13.9 (Microsoft Store, Stable Release)
- Primary Libraries: NumPy, Pandas, & Scikit-Learn (used only for preprocessing utilities and evaluation metrics)
- The ANN and SVM models are implemented using NumPy operations rather than high-level machine learning frameworks. This provides full transparency into the training process.


**Runtime Notes**

- The program typically requires one to two minutes to complete. This is expected due to the following factors:

  1. The ANN uses NumPy for all forward and backward passes, which is slower than GPU-accelerated frameworks.
  2. The SVM performs repeated gradient updates across a high-dimensional one-hot encoded feature space.
  3. Preprocessing steps such as encoding and scaling significantly expand the dataset and require additional computation.
     
- The runtime reflects the dataset size, the number of generated features, and the cost of running iterative training loops in NumPy.


**How to Run the Program**

1. Install Dependencies
   - Ensure that Python 3.13.9 is installed, then install required packages: pip install numpy pandas scikit-learn
2. Prepare the Dataset
   - Place the following files in the project directory: adult.data & adult.test
3. Execute the Program
   - From the terminal: python main.py
4. Program Workflow
   - The script performs the following steps:
   
    1. Loads and cleans the raw training and test data.
    2. Handles missing values for numeric and categorical fields.
    3. Applies one-hot encoding to categorical columns.
    4. Scales all features using standardization.
    5. Trains both the ANN and the SVM.
    6. Evaluates the models using accuracy, precision, recall, F1 score, and confusion matrices.
    7. Saves processed datasets and a model comparison report.


**Output Files**
- The following files are generated in the project directory:

  - adult_preprocessed_train.csv
    - Contains the fully transformed training data after encoding and scaling.
  
  - adult_preprocessed_test.csv
    - Contains the transformed test data in the same format as the training dataset.
  
  - model_comparison_report.txt
    - Includes evaluation metrics and confusion matrices for both the ANN and SVM models.


 
