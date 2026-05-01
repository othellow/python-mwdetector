# Malware Detection System (End-to-End ML + CI/CD)

## Overview
This project is a production-ready machine learning system that classifies software samples as **malware or goodware**.

It includes:
- Model training and evaluation
- Flask web application
- CI/CD pipeline with automated testing
- Live deployment on Render

---

## Live Demo
https://mwdetector-app.onrender.com/

---

## Key Features

### 1. Machine Learning Pipeline
- Stratified 80/20 train-test split
- 10-fold cross-validation
- LightGBM selected as final model

### 2. Performance
- ROC-AUC: **0.9996**
- Accuracy: **0.9972**

---

### 3. Web Application
- Manual feature input (UI form)
- Batch CSV prediction
- Automatic evaluation (AUC, accuracy, confusion matrix)

---

### 4. CI/CD Pipeline
Implemented using GitHub Actions:

- Runs automated tests on every push
- Blocks deployment if tests fail
- Triggers deployment only after success
- Performs post-deploy smoke test (`/health`)

---

## Architecture

```text
User → Flask App → Preprocessor → Model → Prediction
                   ↑
               CI/CD enforced