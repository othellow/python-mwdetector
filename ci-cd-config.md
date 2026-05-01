# CI/CD Configuration

## Overview

This project implements a Continuous Integration and Continuous Deployment (CI/CD) pipeline using GitHub Actions to ensure code reliability and safe deployment of the machine learning web application.

The pipeline enforces automated testing before deployment and verifies the production system after deployment.

---

## CI/CD Workflow Summary

The pipeline follows this sequence:

Push / Pull Request → Run Tests → Deploy → Smoke Test

---

## Tools Used

- GitHub Actions (CI/CD orchestration)
- pytest (testing framework)
- Render (deployment platform)
- curl (for smoke testing)

---

## Trigger Conditions

The CI/CD pipeline is triggered automatically on:

- Push to the `main` branch
- Pull requests targeting the `main` branch

---

## Pipeline Stages

### 1. Environment Setup

The pipeline runs on a clean Linux environment:

- Python version: 3.12
- Dependencies installed from `requirements.txt`

This ensures consistency between local development and production.

---

### 2. Automated Testing (CI)

The pipeline runs automated tests using `pytest`.

#### Types of Tests Implemented:

- **Unit Tests**
  - Validate model wrapper (`predict_malware`)
  - Handle edge cases (empty input, invalid types)

- **Integration Tests**
  - Test API endpoints:
    - `/predict`
    - `/health`

These tests ensure both internal logic and external interfaces function correctly.

---

### 3. Deployment Control

Deployment is triggered only if all tests pass.

- Auto-deploy on Render is disabled
- Deployment is triggered via a secure deploy hook:

curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}

This ensures:
- Broken code is never deployed
- Deployment is controlled and validated

---

### 4. Post-Deployment Smoke Test

After deployment, the pipeline verifies the live system:

GET /health

This confirms:
- The application is running
- The deployment was successful
- The production system is accessible

---

## Key Design Decisions

### Why CI/CD?

- Prevents faulty code from reaching production
- Ensures reproducibility and reliability
- Automates validation steps

---

### Why Smoke Testing?

A successful deployment does not guarantee a working system.

The smoke test ensures:
- The app is live
- The API is responsive
- The deployment did not silently fail

---

### Why GitHub Actions?

- Native integration with GitHub
- Easy configuration
- Supports automated workflows on push/PR

---

## Security Considerations

- Deployment uses a secret (`RENDER_DEPLOY_HOOK`)
- Secrets are stored securely in GitHub repository settings
- Sensitive information is not hardcoded

---

## Limitations

- Deployment verification uses a fixed wait (`sleep 20`), which may be brittle
- No retry logic for smoke tests
- Deployment success is inferred from health check only

---

## Future Improvements

- Replace fixed delay with retry-based health checks
- Add monitoring and alerting
- Track deployment success/failure logs
- Introduce staging environment before production

---

## Conclusion

The CI/CD pipeline ensures that:

- Code is automatically tested before deployment
- Deployment occurs only when tests pass
- The production system is validated after deployment

This transforms the project from a simple ML application into a **reliable, production-ready system with enforced correctness**.