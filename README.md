# End-to-End Driver Churn Prediction Pipeline Using Azure ML and Ensemble Models

---

## 1. Project Overview

In this project, I developed a scalable and accurate driver churn prediction solution for Ola using Azure Machine Learning Studio. The goal was to predict driver attrition based on driver demographics, income, tenure, and performance data from 2019 to 2020, supporting retention and operational strategies.

The solution includes data preprocessing, feature engineering, automated model training with Azure AutoML, ensemble modeling, explainability, and deployment of a real-time inference endpoint.

---

## 2. Data Preparation & Feature Engineering

- Missing values in numerical and binary features were imputed using KNNImputer (k=5).
- Data was aggregated at the driver level (Driver_ID) to create features such as:
  - QuarterlyRating_Increased: Whether the driver’s quarterly rating improved over time.
  - Income_Increased: Whether income increased during the observed period.
- Mode imputation was applied for categorical variables like Gender, City, Education, and Designation.
- The final cleaned and feature-engineered dataset was exported as `driver_attrition_ml_table.csv` for modeling.

---

## 3. Modeling Details

I utilized Azure AutoML to automate model training and selected a weighted soft voting ensemble model based on **highest accuracy**.

| Component              | Details                                                       |
|------------------------|---------------------------------------------------------------|
| Modeling Approach      | Azure AutoML with Voting Ensemble (5 base models ensembled)    |
| Ensemble Strategy     | Weighted soft voting, final model selected on highest accuracy |
| Base Models Used      | XGBoostClassifier, LightGBM (two variants), RandomForest, ExtremeRandomTrees |
| Model Weights         | XGBoost: 0.40, LightGBM: 0.33, RandomForest: 0.13, ExtremeRandomTrees: 0.067 each |
| Best Individual Model | Iteration 1 (XGBoost) with Accuracy: 97.1%                     |
| Pipeline Type         | AutoML-generated pipeline with MeanCrossValidation             |

---

## 4. Model Hyperparameters

| Model             | learning_rate | n_estimators | max_depth | min_child_weight | subsample | colsample_bytree | gamma | num_leaves | min_child_samples | min_samples_split | min_samples_leaf | bootstrap | criterion |
|-------------------|---------------|--------------|-----------|------------------|-----------|------------------|-------|------------|-------------------|-------------------|------------------|-----------|-----------|
| XGBoostClassifier | 0.1           | 100          | 6         | 1                | 0.8       | 0.8              | 0     | –          | –                 | –                 | –                | –         | –         |
| LightGBM_1        | 0.05          | 200          | -1        | –                | 0.7       | 0.9              | –     | 31         | 20                | –                 | –                | –         | –         |
| RandomForest      | –             | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | True      | gini      |
| ExtremeRandomTrees| –             | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | False     | gini      |
| LightGBM_2        | 0.03          | 150          | 10        | –                | 0.8       | 0.8              | –     | 40         | 30                | –                 | –                | –         | –         |

---

## 5. Performance Metrics

| Metric                | Value     |
|-----------------------|-----------|
| Accuracy              | 97.1%     |
| AUC (Macro)           | 0.994     |
| Log Loss              | 0.084     |
| Matthews Corr. Coef.  | 0.936     |
| Precision (Weighted)  | 97.2%     |
| F1 Score (Macro)      | 0.967     |
| Recall (Macro)        | 0.975     |
| Balanced Accuracy     | 97.5%     |
| Sensitivity (Recall)  | 97.1%     |
| Specificity           | (if available) |
| Confusion Matrix      | Available in project reports |

---

## 6. Explainability & Deployment

- Model explainability is enabled using Azure Responsible AI, facilitating interpretation and trust.
- The model is deployed as a real-time serverless inference endpoint with ONNX support for fast scoring.
- A Conda environment (`conda_env.yml`) is provided to ensure reproducibility.
- Deployment artifacts include `model.pkl`, `scoring_file.py`, and the MLTable dataset CSV.

---

## 7. Compute and Infrastructure

- Training was performed on Azure Machine Learning Compute Cluster, providing scalable managed VMs.
- Real-time inference is hosted on a serverless endpoint, allowing automatic scaling and minimal infrastructure management.

---

## 8. Business Insights for Ola

- The high accuracy and recall of the model enable early identification of drivers likely to churn, allowing Ola to implement timely retention measures.
- Drivers showing no income growth or decline in quarterly ratings are at higher risk of churn, suggesting key performance and incentive areas to monitor.
- Proactive churn prediction supports efficient fleet management by anticipating driver availability and reducing operational disruptions.

---

## 9. Repository Contents

- `automl_driver.py` — Training and AutoML pipeline script
- feature-engineering.py - 
- `scoring_file.py` — Scoring script for deployment  
- `conda_env.yml` — Conda environment specification  
- `driver_attrition_ml_table.csv` — Cleaned, feature-engineered dataset (small sample or data schema if dataset excluded)  
- `model.pkl` — Serialized ensemble model artifact  
- `project_report.pdf` — Full project report including detailed results and analysis  
- `README.md` — This file

