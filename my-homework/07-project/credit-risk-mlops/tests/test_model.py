import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from model import prepare_features, predict, COVARIATES


def test_prepare_features_returns_dataframe():
    data = {col: 1.0 for col in COVARIATES}
    result = prepare_features(data)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == COVARIATES


def test_prepare_features_has_one_row():
    data = {col: 0.5 for col in COVARIATES}
    result = prepare_features(data)
    assert len(result) == 1


def test_predict_returns_dict_with_keys():
    mock_model = MagicMock()
    mock_model.predict_partial_hazard.return_value = pd.Series([1.1])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(COVARIATES)))

    features = prepare_features({col: 1.0 for col in COVARIATES})
    result = predict(mock_model, mock_scaler, features)

    assert "partial_hazard" in result
    assert "risk_segment" in result


def test_predict_segment_bajo():
    mock_model = MagicMock()
    mock_model.predict_partial_hazard.return_value = pd.Series([0.5])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(COVARIATES)))

    features = prepare_features({col: 1.0 for col in COVARIATES})
    result = predict(mock_model, mock_scaler, features)
    assert result["risk_segment"] == "bajo"


def test_predict_segment_medio():
    mock_model = MagicMock()
    mock_model.predict_partial_hazard.return_value = pd.Series([1.1])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(COVARIATES)))

    features = prepare_features({col: 1.0 for col in COVARIATES})
    result = predict(mock_model, mock_scaler, features)
    assert result["risk_segment"] == "medio"


def test_predict_segment_alto():
    mock_model = MagicMock()
    mock_model.predict_partial_hazard.return_value = pd.Series([1.5])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(COVARIATES)))

    features = prepare_features({col: 1.0 for col in COVARIATES})
    result = predict(mock_model, mock_scaler, features)
    assert result["risk_segment"] == "alto"


def test_predict_hazard_rounded():
    mock_model = MagicMock()
    mock_model.predict_partial_hazard.return_value = pd.Series([1.23456789])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, len(COVARIATES)))

    features = prepare_features({col: 1.0 for col in COVARIATES})
    result = predict(mock_model, mock_scaler, features)
    assert result["partial_hazard"] == 1.2346
