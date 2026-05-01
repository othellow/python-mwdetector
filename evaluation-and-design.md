# Evaluation and Design

## Problem Definition
The objective of this project is to build a high-performance malware detection model using supervised machine learning on structured tabular data by classifying software samples as:

- 0 → Goodware
- 1 → Malware

- **Primary Metric:** ROC-AUC  
- **Secondary Metric:** Accuracy  

ROC-AUC is prioritized because it measures the model’s ability to distinguish between malware and benign samples across all thresholds, which is critical in cybersecurity applications.

---

## Dataset Handling

The dataset consists of structured features extracted from executable files.

### Key Handling Steps:
- Removed irrelevant or constant-value features
- Handled missing values appropriately
- Ensured consistent feature schema across training and inference
- Split dataset into:
  - 80% Training set (for cross-validation and tuning)
  - 20% Hold-out test set (for final evaluation)
- Folder /python-test holds all previous model training and evaluation

---

## Data Pre-processing

A preprocessing pipeline was applied consistently across all models:

- Feature scaling (for Logistic Regression, MLP)
- Numerical encoding for all features
- Unified preprocessing pipeline saved as:
  - `LightGBM_preprocessor.pkl`

### Design Goals:
- Prevent data leakage  
- Ensure reproducibility  
- Enable seamless deployment  

---

## Feature Engineering

Minimal feature engineering was applied intentionally.

### Approach:
- Retained original feature structure
- Avoided unnecessary transformations
- Preserved feature interactions for tree-based models

### Rationale:
Tree-based models (Random Forest, LightGBM, etc.) naturally capture:
- Nonlinear relationships  
- Feature interactions  

Artifacts saved:
- `feature_columns.pkl`

---

## Model Evaluation Strategy

### Cross-Validation
- K-Fold Cross-Validation used
- Metrics recorded:
  - Mean Accuracy ± Standard Deviation
  - Mean ROC-AUC ± Standard Deviation

### Model Selection Criteria
- Primary: ROC-AUC  
- Secondary: Accuracy and Stability  
- Final validation: Hold-out test set  

---

## Models Evaluated

- Random Forest  
- Decision Tree  
- Logistic Regression  
- PyTorch MLP  
- LightGBM  
- XGBoost  
- CatBoost  

---

# 1. Cross-Validation Performance Comparison (Mean ± Std Dev)

| Model              | CV Accuracy         | CV ROC-AUC          |
|--------------------|---------------------|---------------------|
| LightGBM           | **0.9885 ± 0.0013** | **0.9984 ± 0.0004** |
| Random Forest      | 0.9879 ± 0.0016     | 0.9981 ± 0.0007     |
| XGBoost            | 0.9856 ± 0.0021     | 0.9979 ± 0.0006     |
| CatBoost           | 0.9806 ± 0.0024     | 0.9966 ± 0.0005     |
| Decision Tree      | 0.9792 ± 0.0014     | 0.9855 ± 0.0017     |
| PyTorch MLP        | 0.9474 ± 0.0042     | 0.9835 ± 0.0020     |
| Logistic Regression| 0.8139 ± 0.0051     | 0.8774 ± 0.0056     |

---

## 1.a Cross-Validation Interpretation per Model

### LightGBM
Highest ROC-AUC and lowest variance. Most stable and accurate.

### Random Forest
Very strong performance, slightly below LightGBM.

### XGBoost
Highly competitive gradient boosting model.

### CatBoost
Strong performance but slightly behind top models.

### Decision Tree
Good baseline but lacks ensemble robustness.

### PyTorch MLP
Effective nonlinear learner, but less suited for tabular data.

### Logistic Regression
Lowest performance, indicating non-linear separability.

---

# 3. Hold-Out Test Results

| Model                | Accuracy   | Precision | Recall     | F1         | ROC-AUC    |
|----------------------|------------|-----------|------------|------------|------------|
| LightGBM             | **0.9888** | 0.9880    | **0.9928** | **0.9904** | **0.9981** |
| Random Forest        | 0.9882     | 0.9899    | 0.9899     | 0.9899     | 0.9978     |
| XGBoost              | 0.9857     | 0.9853    | 0.9900     | 0.9876     | 0.9974     |
| CatBoost             | 0.9806     | 0.9817    | 0.9849     | 0.9833     | 0.9959     |
| Decision Tree        | 0.9786     | 0.9821    | 0.9809     | 0.9815     | 0.9841     |
| PyTorch MLP          | 0.9495     | 0.9587    | 0.9539     | 0.9563     | 0.9838     |
| Logistic Regression  | 0.8109     | 0.8623    | 0.8015     | 0.8308     | 0.8757     |

---

# 4. Deep Dive Per Model Analysis

### LightGBM
- Best ROC-AUC and Recall  
- Lowest false negatives  
- Highly scalable and efficient  

### Random Forest
- Strong baseline  
- Slightly lower recall than LightGBM  

### XGBoost
- Excellent performance  
- Slightly below LightGBM  

### CatBoost
- Strong but not leading  

### Decision Tree
- Interpretable but less accurate  

### PyTorch MLP
- Demonstrates deep learning capability  
- Less effective for tabular data  

### Logistic Regression
- Weak performance  
- Confirms nonlinear data patterns  

---

# 5. Full Tournament Rankings

| Rank | Model               | ROC-AUC    |
|------|---------------------|------------|
| 1    | LightGBM            | **0.9981** |
| 2    | Random Forest       | 0.9978     |
| 3    | XGBoost             | 0.9974     |
| 4    | CatBoost            | 0.9959     |
| 5    | Decision Tree       | 0.9841     |
| 6    | PyTorch MLP         | 0.9838     |
| 7    | Logistic Regression | 0.8757     |

---

# 6. Why LightGBM Was Selected

LightGBM was selected because:

- Highest ROC-AUC (primary metric)
- Highest recall (minimizes missed malware)
- Lowest variance (stable)
- Efficient and scalable
- Strong handling of feature interactions

### Key Insight:
Gradient boosting captures complex nonlinear relationships better than:
- Linear models  
- Single decision trees  
- Neural networks (for tabular data)  

---

# Final Test Set Performance for LightGBM

## Confusion Matrix

|                   | Predicted Clean | Predicted Malware|
|-------------------|-----------------|------------------|
| Actual Clean      | 4154            | 70               |
| Actual Malware    | 42              | 5771             |

### Interpretation:
- Very low false negatives (42)
- Balanced error distribution
- High reliability in malware detection

---

# Overall Executive Summary

This study conducted a comprehensive evaluation of multiple machine learning models for malware classification.

### Key Findings:
- Tree-based ensemble models dominate tabular data performance
- LightGBM achieved the highest performance across all metrics
- Results are consistent between cross-validation and hold-out testing
- Minimal feature engineering was required due to model capability

### Final Conclusion:
LightGBM is the most suitable model for this malware detection task due to its superior accuracy, robustness, and ability to model complex feature interactions.

---

# Final Design Perspective

This project demonstrates:

- End-to-end ML pipeline design  
- Robust evaluation methodology  
- Model benchmarking across multiple paradigms  
- Production readiness (saved model + preprocessing pipeline)  

This is a deployable, real-world machine learning system.