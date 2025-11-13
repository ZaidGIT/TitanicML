# Titanic Survival Prediction Model

A data science project focused on predicting the survival of passengers from the RMS Titanic using a tuned **Random Forest Classifier**. This project establishes a complete machine learning pipeline, emphasizing data integrity, feature engineering, and hyperparameter optimization.

## Project Overview

The objective of this project is to develop a predictive model that accurately classifies survival outcomes (`Survived`: 1 for Yes, 0 for No) using the dataset provided by the Kaggle Titanic - Machine Learning from Disaster competition.

### Key Methodology and Findings:

* **Feature Engineering:** Features were derived from high-value categorical data, including the **Deck** (extracted from `Cabin`) and the **Title** (extracted from `Name`).
* **Data Integrity:** Imputation for missing values (`Age`, `Fare`, `Embarked`) was performed strictly using statistics derived **only from the training set** to prevent data leakage into the test set.
* **Preprocessing:** All categorical features were handled via **One-Hot Encoding**, and numerical features were standardized using **StandardScaler** to ensure equal contribution to the model.
* **Model Optimization:** The model was tuned using **10-Fold Cross-Validation** to find optimal hyperparameters, successfully regularizing the model (Training Accuracy reduced from an initial 100% to 92.26%).

---

## Installation and Setup

This project uses **Poetry** for efficient dependency management and creation of an isolated virtual environment.

### Prerequisites

You must have **Python 3.x** and the **Poetry** package manager installed on your system.

```bash
# Install Poetry globally (if necessary)
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python -