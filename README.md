# 💳 Credit Card Default Risk Prediction System

## 📌 Project Overview

The **Credit Card Default Risk Prediction System** is a Machine Learning project designed to predict whether a credit card customer is likely to default on their payment in the next month.

The project uses customer demographic information, credit limits, repayment history, bill statements, and payment amounts to identify potential default risks. Multiple machine learning classification algorithms are implemented, tuned, and evaluated to compare their predictive performance.

The project also explores the effect of class imbalance by conducting experiments with the original dataset and a downsampled dataset.

## 🎯 Objectives

* Analyze credit card customer data to identify patterns associated with payment defaults.
* Perform data cleaning, exploratory data analysis (EDA), and feature engineering.
* Handle imbalanced target classes.
* Build and compare multiple machine learning classification models.
* Optimize model hyperparameters using GridSearchCV and cross-validation.
* Evaluate model performance using Accuracy, Precision, Recall, and F1-score.
* Select and save a trained model for future predictions.

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Environment:** Jupyter Notebook / Google Colab
* **Data Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
* **Model Optimization:** GridSearchCV, Cross-validation
* **Model Persistence:** Pickle

## 📂 Project Structure

```text
Credit-Card-Default-Risk-Prediction/
│
├── Credit_Card_Default_Risk_Prediction_System.ipynb
├── model.pkl
└── README.md
```

*Note: The dataset and trained model should be added to the repository only if you have permission to share them. The `model.pkl` file is generated after training.*

## 📊 Dataset Description

The project uses a credit card customer dataset containing demographic details, credit information, repayment status, bill amounts, and payment history.

Some of the important features include:

* `LIMIT_BAL` – Amount of credit granted to the customer.
* `SEX` – Gender of the customer.
* `EDUCATION` – Education level.
* `MARRIAGE` – Marital status.
* `AGE` – Age of the customer.
* `PAY_0` to `PAY_6` – Historical repayment status.
* `BILL_AMT1` to `BILL_AMT6` – Monthly bill statement amounts.
* `PAY_AMT1` to `PAY_AMT6` – Previous payment amounts.
* `default` – Target variable indicating whether the customer defaults.

The target variable is encoded as:

* `1` – Default
* `0` – No default

## 🔍 Project Workflow

### 1. Data Loading and Cleaning

* Load the dataset using Pandas.
* Inspect data types, dimensions, and statistical summaries.
* Check for missing values and duplicate records.
* Examine categorical values and target class distribution.
* Clean and prepare the dataset for further analysis.

### 2. Exploratory Data Analysis (EDA)

Exploratory data analysis is performed to understand customer characteristics and discover patterns related to credit card defaults.

The analysis includes:

* Default and non-default customer distribution.
* Customer age and education distribution.
* Credit limit analysis.
* Default rates across education and gender groups.
* Repayment delays and their relationship with default.
* Monthly bill amounts and payment patterns.
* Correlation analysis between numerical features.

Visualizations are created using Matplotlib and Seaborn.

### 3. Feature Engineering

New features are created to capture additional information about customer repayment behavior.

* `Total_Bill` – Sum of monthly bill amounts.
* `Total_Payment` – Sum of monthly payments.
* `DELAYED_MONTHS` – Number of months with delayed repayment.
* `MAX_REPAYMENT_DELAY` – Maximum repayment delay status.
* `AVG_REPAYMENT_STATUS` – Average repayment status.
* `PAYMENT_BILL_RATIO` – Ratio of total payments to total bills.

Redundant and highly correlated features are removed before model training.

### 4. Data Preprocessing

The dataset is divided into independent features (`X`) and the target variable (`Y`).

Preprocessing techniques include:

* **StandardScaler:** Standardizes numerical features.
* **OneHotEncoder:** Encodes categorical features.
* **ColumnTransformer:** Applies the appropriate transformations to numerical and categorical columns.
* **Train-Test Split:** Divides the dataset into 80% training data and 20% testing data, using stratification to preserve class proportions.

### 5. Model Building

Seven machine learning classification algorithms are implemented and evaluated:

1. K-Nearest Neighbors (KNN)
2. Naive Bayes
3. Decision Tree
4. Logistic Regression
5. Support Vector Machine (SVM)
6. Random Forest
7. AdaBoost

Hyperparameter tuning is performed using GridSearchCV with cross-validation and F1-score as the scoring metric.

### 6. Handling Imbalanced Data

Since the target variable is imbalanced, a second experiment is conducted using downsampling.

The performance of the classification models is evaluated on the downsampled dataset and compared with the results from the original dataset.

### 7. Model Evaluation

The models are evaluated using the following performance metrics:

| Metric           | Description                                               |
| ---------------- | --------------------------------------------------------- |
| Accuracy         | Overall proportion of correct predictions                 |
| Precision        | Proportion of predicted defaults that are actual defaults |
| Recall           | Proportion of actual defaults correctly identified        |
| F1-score         | Harmonic mean of precision and recall                     |
| Confusion Matrix | Summary of correct and incorrect classifications          |

F1-score is emphasized because the target variable is imbalanced and both precision and recall are important for identifying potential defaulters.

## 📈 Results

In the original dataset experiment, the notebook reports the following results for selected models:

| Model         |           Accuracy | F1-score |
| ------------- | -----------------: | -------: |
| KNN           |             81.02% |   0.4635 |
| Random Forest | Not specified here |   0.5426 |

Random Forest was selected as the final model in the original dataset experiment based on its reported F1-score of 0.5426.

The notebook also includes a second experiment using downsampling, with evaluation metrics calculated for each model.

## 💾 Model Saving

The selected Random Forest estimator is saved using Python's Pickle library.

```python
import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(grid_rf.best_estimator_, file)
```

The saved model can be loaded later for further experimentation or prediction, provided the same compatible preprocessing and input feature structure are used.

## 🚀 How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Credit-Card-Default-Risk-Prediction.git
```

### Step 2: Navigate to the project directory

```bash
cd Credit-Card-Default-Risk-Prediction
```

### Step 3: Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 4: Add the dataset

Place the dataset file in the project directory and update the dataset path in the notebook if necessary.

### Step 5: Open the Jupyter Notebook

```bash
jupyter notebook
```

Open `Credit_Card_Default_Risk_Prediction_System.ipynb` and execute the cells in sequence.

Alternatively, upload the notebook and dataset to Google Colab and run the project there.

## 🔮 Future Enhancements

* Develop a web application for interactive credit default prediction.
* Add a user-friendly interface for entering customer information.
* Explore additional techniques for handling imbalanced data.
* Perform more extensive feature selection and model optimization.
* Integrate model monitoring and retraining capabilities.
* Deploy the trained model using Flask, Django, or Streamlit. 
