# Project Summary

---

## 1. Project Overview & Business Objective

In this project, I aimed to build a scalable and accurate driver churn prediction solution for Ola. Using Azure Machine Learning Studio, I leveraged managed compute clusters for training and designed a pipeline ready for deployment on serverless real-time endpoints to support proactive retention strategies.

Key points about the project and its business objective:

- Predict driver attrition to enable timely interventions.
- Dataset includes driver demographics, income, tenure, and performance data from 2019 to 2020.
- The solution is deployment-ready but has not been deployed to avoid incurring unnecessary costs.

| Aspect             | Details                                                         |
|--------------------|-----------------------------------------------------------------|
| Business Objective  | Predict driver attrition (churn) to support retention strategies at Ola |
| Platform           | Azure Machine Learning Studio (AutoML + Designer)               |
| Dataset Overview   | Driver records from 2019–2020, including demographics, income, dates, performance |
| Target Variable    | Attrition (1 if driver left, based on non-null LastWorkingDate; else 0) |

---

## 2. Data Preparation & Feature Engineering

To ensure data quality and extract meaningful insights, I:

- Used KNNImputer (k=5) for imputing missing numerical and binary data.
- Aggregated data at the Driver_ID level to create driver-specific features.
- Engineered features such as QuarterlyRating_Increased and Income_Increased to capture trends over time.
- Applied mode imputation for categorical fields like Gender, City, Education, and Designation.
- Exported the final cleaned and feature-engineered dataset for modeling.

| Step                | Details                                                          |
|---------------------|------------------------------------------------------------------|
| Imputation Strategy | KNNImputer (k=5) for numerical and binary features              |
| Feature Engineering | Aggregation at Driver_ID level                                   |
|                     | - QuarterlyRating_Increased: any improvement over time          |
|                     | - Income_Increased: income growth over months                    |
|                     | - Mode imputation for Gender, City, Education, Designation      |
|                     | - Reporting_Date = max, Dateofjoining = min                      |
| Final Dataset       | Cleaned, feature-engineered dataset exported as `driver_attrition_ml_table.csv` |

---

## 3. Modeling Details

I automated model training with Azure AutoML and enhanced predictive power through ensemble learning:

- Combined five base models using weighted soft voting, selecting the final model by highest accuracy.
- Base models: XGBoost, two LightGBM variants, Random Forest, Extremely Randomized Trees.
- Best individual model: XGBoost with 97.1% accuracy.
- Used cross-validation to validate model stability.

| Component              | Details                                                       |
|------------------------|---------------------------------------------------------------|
| Modeling Approach      | Azure AutoML with Voting Ensemble (5 base models ensembled)    |
| Ensemble Strategy     | Weighted soft voting, final selection based on accuracy        |
| Base Models Used      | XGBoostClassifier, LightGBM (x2), RandomForest, ExtremeRandomTrees |
| Model Weights         | XGBoost: 0.40, LightGBM: 0.33, RandomForest: 0.13, ExtremeRandomTrees: 0.067 each |
| Best Individual Model | Iteration 1 (XGBoost) with Accuracy: 97.1%                     |
| Pipeline Type         | AutoML-generated pipeline with MeanCrossValidation             |

---

## 4. Model Hyperparameters

Key hyperparameters tuned for the top base models:

| Model            | learning_rate | n_estimators | max_depth | min_child_weight | subsample | colsample_bytree | gamma | num_leaves | min_child_samples | min_samples_split | min_samples_leaf | bootstrap | criterion |
|------------------|---------------|--------------|-----------|------------------|-----------|------------------|-------|------------|-------------------|-------------------|------------------|-----------|-----------|
| XGBoostClassifier | 0.1          | 100          | 6         | 1                | 0.8       | 0.8              | 0     | –          | –                 | –                 | –                | –         | –         |
| LightGBM_1        | 0.05         | 200          | -1        | –                | 0.7       | 0.9              | –     | 31         | 20                | –                 | –                | –         | –         |
| RandomForest      | –            | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | True      | gini      |
| ExtremeRandomTrees| –            | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | False     | gini      |
| LightGBM_2        | 0.03         | 150          | 10        | –                | 0.8       | 0.8              | –     | 40         | 30                | –                 | –                | –         | –         |

---

## 5. Performance Metrics

The ensemble model achieved strong predictive results:

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
| Specificity           | (calculated if available) |
| Confusion Matrix      | Available in project reports |

---

## 6. Explainability & Deployment Readiness

- Enabled model explainability using Azure Responsible AI for transparency and trust.
- The pipeline and artifacts (`conda_env.yml`, `model.pkl`, scoring script) are fully prepared for deployment.
- To avoid unnecessary expenses, I have not deployed the model yet; however, it can be deployed on Azure ML real-time endpoints seamlessly as needed.

---

## 7. Compute and Infrastructure

- Training used Azure Machine Learning Compute Cluster (scalable managed VMs).
- The solution supports deployment on serverless real-time inference endpoints with automatic scaling and minimal maintenance.

---

## 8. Business Insights for Ola

Based on model findings, key actionable insights include:

- Early identification of drivers likely to churn enables timely retention incentives and support.
- Drivers with no income growth or declining quarterly ratings are higher risk, highlighting areas for performance improvement and engagement.
- Predictive churn estimates support efficient resource planning, reducing driver shortages and improving customer service continuity.

---

# End of Summary
