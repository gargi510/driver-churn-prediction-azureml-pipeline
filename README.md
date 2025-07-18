# End-to-End Driver Churn Prediction Pipeline Using Azure ML and Ensemble Models

## Project Overview  
This project implements a complete machine learning pipeline to predict driver churn for Ola using Azure Machine Learning Studio. The solution includes data preprocessing, feature engineering, model training with ensemble methods, hyperparameter tuning, and deployment-ready components. It leverages Azure AutoML and custom pipelines to achieve high accuracy and robust performance for proactive driver retention.

## Features & Highlights  
- Robust feature engineering and missing value imputation using KNN  
- Ensemble modeling combining XGBoost, LightGBM, Random Forest, and other classifiers  
- Automated hyperparameter tuning and early stopping in Azure ML AutoML  
- Modular pipeline creation with Azure ML designer components  
- Real-time model deployment support with inference endpoints  
- Integration with GitHub for version control and reproducibility  

## Dataset  
Due to privacy concerns, the proprietary driver dataset is **not included** in this repository. Please contact the project owner for access or use your own relevant data. Sample data schemas and preprocessing scripts are provided to assist data preparation.

## Model Artifacts  
Trained model files (`model.pkl`) are **excluded** from this repository to ensure repository cleanliness and security. All training and deployment scripts are included to enable model reproduction. Models are version-controlled and stored securely in Azure ML Model Registry.

## Results Summary  
The ensemble model pipeline achieved strong performance metrics including:  
- Accuracy: ~97.1%  
- Weighted AUC: ~99.4%  
- F1 Score (weighted): ~97.1%  
- Balanced Accuracy: ~97.5%  

For a detailed project summary, full metric tables, and insights, please see the [PROJECT_REPORT.md](./PROJECT_REPORT.md) file.

## Getting Started

### Prerequisites  
- Python 3.7+  
- Azure ML SDK  
- Required Python packages (install via `requirements.txt`)  
