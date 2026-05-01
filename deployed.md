# Deployed Application

## Live Web Application
https://mwdetector-app.onrender.com/

## Health Check Endpoint
https://mwdetector-app.onrender.com/health

## Description
This application is a malware detection system built using Flask and a trained LightGBM model.

### Features:
- Manual prediction via UI form
- Batch CSV upload for multiple predictions
- Automatic evaluation (AUC, accuracy, confusion matrix) when labels are provided
- Health endpoint for monitoring and CI/CD smoke testing

## Deployment Details
- Platform: Render (free-tier)
- Deployment Trigger: GitHub Actions (CI/CD pipeline)
- Deployment Condition: Only after tests pass

## Notes
- The application uses pre-trained model artifacts (`model.pkl`, `preprocessor.pkl`, `feature_columns.pkl`)
- CI/CD pipeline ensures production stability before deployment