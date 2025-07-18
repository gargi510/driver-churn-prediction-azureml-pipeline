# Project Summary

---

## 1. Project Overview & Business Objective

In this project, I aimed to build a scalable and accurate driver churn prediction solution for Ola. Using Azure Machine Learning Studio, I leveraged managed compute clusters for training and serverless endpoints for real-time inference to support proactive retention strategies.

Some key points about the project and its business objective:

- I focused on predicting driver attrition to enable timely interventions.
- The dataset covered driver demographics, income, tenure, and performance data from 2019 to 2020.
- My goal was to deploy a solution that supports real-time inference and is fully reproducible.

| Aspect             | Details                                                         |
|--------------------|-----------------------------------------------------------------|
| Business Objective  | Predict driver attrition (churn) to support retention strategies at Ola |
| Platform           | Azure Machine Learning Studio (AutoML + Designer)               |
| Dataset Overview   | Driver records from 2019–2020, including demographics, income, dates, performance |
| Target Variable    | Attrition (1 if driver left, based on non-null LastWorkingDate; else 0) |

---

## 2. Data Preparation & Feature Engineering

I ensured data quality and created meaningful features through the following steps:

- I used KNNImputer (k=5) to impute missing values for numerical and binary fields.
- I aggregated data at the Driver_ID level to create driver-specific features.
- I engineered features indicating whether quarterly ratings and income increased over time.
- For categorical variables like Gender, City, and Designation, I applied mode imputation.
- I exported the cleaned and feature-engineered dataset for modeling.

| Step                | Details                                                          |
|---------------------|------------------------------------------------------------------|
| Imputation Strategy | KNNImputer (k=5) for numerical and binary features              |
| Feature Engineering | Aggregation at Driver_ID level                                   |
|                     | - QuarterlyRating_Increased: any improvement over time          |
|                     | - Income_Increased: income growth over months                    |
|                     | - Mode used for Gender, City, Education, Designation            |
|                     | - Reporting_Date = max, Dateofjoining = min                      |
| Final Dataset       | Cleaned, feature-engineered dataset exported as `driver_attrition_ml_table.csv` |

---

## 3. Modeling Details

I leveraged Azure AutoML to automate model training and used an ensemble approach to boost accuracy:

- I ensembled five base models using weighted soft voting, selecting the final model based on highest accuracy.
- The base models included XGBoost, two LightGBM variants, Random Forest, and Extremely Randomized Trees.
- The best individual model was XGBoost, achieving the highest accuracy of 97.1%.
- I used cross-validation to ensure the model’s robustness.

| Component              | Details                                                       |
|------------------------|---------------------------------------------------------------|
| Modeling Approach      | Azure AutoML with Voting Ensemble (5 iterations ensembled)     |
| Ensemble Strategy     | Weighted soft voting, final selection based on accuracy        |
| Base Models Used      | XGBoostClassifier, LightGBM (x2), RandomForest, ExtremeRandomTrees |
| Model Weights         | XGBoost: 0.40, LightGBM: 0.33, RandomForest: 0.13, ExtremeRandomTrees: 0.067 each |
| Best Individual Model | Iteration 1 (XGBoost) with Accuracy: 97.1%                     |
| Pipeline Type         | AutoML-generated pipeline with MeanCrossValidation             |

---

## 4. Model Hyperparameters

Here are the main hyperparameters I tuned for the top base models in the ensemble:

| Model            | learning_rate | n_estimators | max_depth | min_child_weight | subsample | colsample_bytree | gamma | num_leaves | min_child_samples | min_samples_split | min_samples_leaf | bootstrap | criterion |
|------------------|---------------|--------------|-----------|------------------|-----------|------------------|-------|------------|-------------------|-------------------|------------------|-----------|-----------|
| XGBoostClassifier | 0.1          | 100          | 6         | 1                | 0.8       | 0.8              | 0     | –          | –                 | –                 | –                | –         | –         |
| LightGBM_1        | 0.05         | 200          | -1        | –                | 0.7       | 0.9              | –     | 31         | 20                | –                 | –                | –         | –         |
| RandomForest      | –            | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | True      | gini      |
| ExtremeRandomTrees| –            | 100          | None      | –                | –         | –                | –     | –          | –                 | 2                 | 1                | False     | gini      |
| LightGBM_2        | 0.03         | 150          | 10        | –                | 0.8       | 0.8              | –     | 40         | 30                | –                 | –                | –         | –         |

---

## 5. Performance Metrics

My ensemble model delivered strong predictive performance:

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

## 6. Explainability & Deployment

To ensure transparency and operational readiness:

- I enabled model explainability via Azure Responsible AI.
- I deployed a real-time scoring endpoint with ONNX support.
- The Conda environment was defined to ensure reproducibility.
- I saved all artifacts needed for deployment and ongoing maintenance.

| Component         | Details                                              |
|-------------------|------------------------------------------------------|
| Explainability    | Enabled via Azure Responsible AI (`model_explain_run_id`) |
| Deployment        | Real-time serverless inference endpoint with scoring script and ONNX support |
| Environment       | Conda-based, fully reproducible (`conda_env.yml`)    |
| Model Artifacts   | `model.pkl`, `scoring_file.py`, MLTable dataset CSV  |

---

## 7. Compute and Deployment

- **Training Compute:** Azure Machine Learning Compute Cluster (managed scalable VMs)  
- **Inference Deployment:** Serverless real-time inference endpoint, enabling auto-scaling without infrastructure management

---

## 8. Business Insights for Ola

Based on the model’s performance and feature importance, I derived actionable insights to help Ola reduce driver churn and improve operational efficiency:

### Retention Strategies
- The high accuracy and recall enable timely identification of drivers at risk of attrition, allowing proactive engagement through incentives or support.
- Drivers with declining quarterly ratings or income without increase over time are more likely to churn, suggesting targeted interventions for performance and earnings growth.

### Resource Planning
- Predictive churn estimates allow Ola to better forecast driver availability, optimizing fleet size and reducing service gaps.
- Monitoring income and performance trends can help design personalized retention programs, improving driver satisfaction.

### Risk Mitigation
- Early detection of at-risk drivers reduces sudden drops in driver supply, ensuring a stable service experience for customers.
- Deployment of the real-time model means retention teams get up-to-date risk scores, enabling agile decision-making.

---

# End of Summary

