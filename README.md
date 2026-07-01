# Bank Personal Loan Prediction

## Project Overview

This project predicts whether a bank customer is likely to accept a personal loan offer using demographic, financial, and banking-behaviour data. The work is completed in two phases: exploratory data analysis and preprocessing in Phase 1, followed by predictive modelling, hyperparameter tuning, and model comparison in Phase 2.

The project demonstrates an end-to-end machine learning workflow, including data cleaning, feature analysis, feature selection, supervised classification, cross-validation, and model performance evaluation.

## Business Problem

Banks often run personal loan marketing campaigns, but contacting every customer can be inefficient and costly. A predictive model can help identify customers who are more likely to accept a loan offer, allowing the bank to improve campaign targeting, reduce unnecessary outreach, and increase conversion rates.

## Dataset

The project uses the **Bank Personal Loan Modelling** dataset, originally sourced from Kaggle. The dataset contains customer-level information such as age, income, experience, family size, education level, credit card spending, mortgage value, and banking product usage.

The target variable is:

- `Personal_Loan`
  - `1`: Customer accepted the personal loan offer
  - `0`: Customer did not accept the personal loan offer

## Project Structure

```text
bank-personal-loan-prediction
│   README.md
│   requirements.txt
│   .gitignore
│
├── data
│   └── Bank_Personal_Loan_Modelling.csv
│
├── notebooks
│   ├── Phase1_Group71.ipynb
│   └── Phase2_Group71.ipynb
│
├── reports
│   ├── Phase1_Group71.html
│   └── Phase2_Group71.html
│
└── docs
    └── project_structure.txt
```

## Phase 1: Data Preparation and Visualisation

Phase 1 focuses on understanding the dataset and preparing it for modelling.

### Key Steps

- Imported and inspected the Bank Personal Loan dataset
- Reviewed feature names, data types, and feature meanings
- Removed irrelevant columns that did not support analysis
- Renamed columns for better readability and consistency
- Checked unique values across features
- Converted `CreditCard_Spending` into a numeric format
- Removed invalid records where `Experience` had negative values
- Checked for missing values and duplicate observations
- Performed exploratory data analysis using visualisations

### Exploratory Data Analysis

The analysis explored one-variable, two-variable, and three-variable relationships.

Important patterns identified include:

- Customers with higher annual income were more likely to accept a personal loan.
- Customers with higher credit card spending showed stronger loan acceptance behaviour.
- Education level appeared to influence loan acceptance.
- Customers with larger mortgage values were more likely to accept loan offers.
- The target variable was imbalanced, with more customers rejecting the loan than accepting it.

## Phase 2: Predictive Modelling

Phase 2 focuses on building and evaluating machine learning models to predict personal loan acceptance.

### Preprocessing

The cleaned dataset from Phase 1 was used for model development. The preprocessing workflow included:

- Dropping the customer identifier column
- Separating features from the target variable
- Scaling features where required using `StandardScaler`
- Splitting the dataset into training and testing sets
- Applying stratified sampling to preserve target class distribution

## Feature Selection

Feature selection was performed using `SelectKBest` with the ANOVA F-test method (`f_classif`). The top 10 features were selected based on their F-scores.

### Top Selected Features

The strongest predictors included:

1. Annual Income
2. Credit Card Spending
3. CD Account
4. Mortgage
5. Education Level
6. Family Size
7. Securities Account
8. Age
9. Experience
10. Online Banking

These selected features were used consistently across the machine learning models for fair comparison.

## Models Used

The following supervised machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- k-Nearest Neighbors
- Neural Network using TensorFlow/Keras

## Hyperparameter Tuning

Model tuning was performed using `GridSearchCV` for traditional machine learning models and a manual tuning approach for the neural network.

### Tuned Models

#### Random Forest

The Random Forest model was tuned using:

- Number of estimators
- Maximum tree depth
- Minimum samples split
- Minimum samples leaf
- Maximum features

Best parameters:

```text
n_estimators = 150
max_depth = 10
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
```

#### Logistic Regression

Logistic Regression was tuned using:

- Regularisation penalty
- Regularisation strength `C`
- Solver

#### Decision Tree

Decision Tree was tuned using:

- Criterion
- Maximum depth
- Minimum samples split
- Minimum samples leaf

#### Support Vector Machine

SVM was tuned using:

- Kernel type
- Regularisation parameter `C`
- Gamma

#### k-Nearest Neighbors

KNN was tuned using:

- Number of neighbors
- Distance metric
- Weighting strategy

#### Neural Network

The neural network was built using TensorFlow/Keras and tuned using:

- Number of hidden layers
- Number of neurons
- Learning rate
- Dropout rate
- Batch size

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- 5-fold cross-validation
- Paired t-tests for model comparison

ROC-AUC was used as the main comparison metric because it is suitable for binary classification and provides a threshold-independent performance measure.

## Model Performance

The 5-fold cross-validation ROC-AUC scores were:

| Model | Mean ROC-AUC |
|---|---:|
| Random Forest | 0.9977 |
| Decision Tree | 0.9931 |
| Support Vector Machine | 0.9850 |
| k-Nearest Neighbors | 0.9713 |
| Logistic Regression | 0.9559 |

The Random Forest model achieved the best overall performance and statistically outperformed the other models in paired t-tests.

## Best Model

The best-performing model was the **Random Forest Classifier**.

It achieved:

- Test ROC-AUC: approximately **0.9984**
- High precision and recall for both classes
- Strong generalisation under 5-fold cross-validation
- Better performance than Logistic Regression, Decision Tree, SVM, and KNN

The results suggest that ensemble-based non-linear models are highly effective for predicting personal loan acceptance from mixed customer demographic and financial data.

## Key Findings

- Annual income was the strongest predictor of personal loan acceptance.
- Credit card spending and CD account ownership were also highly important.
- Non-linear models performed better than linear models.
- Random Forest provided the strongest balance between predictive performance and robustness.
- Logistic Regression performed the weakest, suggesting that a simple linear decision boundary was not sufficient for this dataset.
- The target variable was imbalanced, which should be considered in future improvements.

## Limitations

- The dataset may not fully represent all real-world banking customers.
- Class imbalance was identified but not explicitly handled using techniques such as SMOTE or class weighting.
- The neural network tuning process was manual and sequential rather than fully automated.
- More advanced ensemble methods such as stacking or blending were not tested.
- Highly accurate models such as Random Forest and Neural Networks may be less interpretable than simpler models.

## Future Improvements

Future work could include:

- Handling class imbalance using SMOTE, undersampling, oversampling, or class weights
- Adding model explainability using SHAP or feature importance analysis
- Testing advanced boosting models such as XGBoost, LightGBM, or CatBoost
- Building a Streamlit dashboard for interactive prediction
- Saving the best model as a reusable `.pkl` or `.joblib` file
- Creating an API endpoint for real-time loan acceptance prediction

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- TensorFlow/Keras
- Tabulate

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/bank-personal-loan-prediction.git
```

Navigate into the project folder:

```bash
cd bank-personal-loan-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks from the `notebooks/` folder.

## Requirements

A suggested `requirements.txt` for this project is:

```text
pandas
matplotlib
seaborn
scikit-learn
scipy
tensorflow
tabulate
jupyter
notebook
ipykernel
```

## Reports

HTML exports of both project phases are available in the `reports/` folder:

- `Phase1_Group71.html`
- `Phase2_Group71.html`

These files allow the project analysis to be viewed without running the notebooks.

## Contributors

- Karan Naresh Rathod
- Shrey Dineshchandra Deshmukh
- Somya Sharma

## Author Note

This repository is prepared as a portfolio-ready machine learning project demonstrating data preprocessing, exploratory data analysis, feature selection, classification modelling, hyperparameter tuning, and model evaluation for a real-world banking marketing use case.
