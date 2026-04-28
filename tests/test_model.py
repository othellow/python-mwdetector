# tests/test_model.py

import pytest

# Import your model loading + prediction logic
# Adjust this import based on your actual structure
from app import predict_malware  # <-- we may adjust this if needed


def test_model_prediction():
    """
    Purpose:
    Ensure the model can take input and return a valid prediction.
    This prevents silent model loading or inference failures.
    """

    # Empty input is acceptable; app prediction fills defaults.
    sample_input = {}

    result = predict_malware(sample_input)

    # Basic assertions
    assert result is not None
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence" in result