import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import prepare_features, predict
from unittest.mock import MagicMock
import numpy as np


def test_prepare_features():
    features = prepare_features("151", "239", 2.5)
    assert features["PU_DO"] == "151_239"
    assert features["trip_distance"] == 2.5


def test_prepare_features_combines_locations():
    features = prepare_features("10", "20", 1.0)
    assert features["PU_DO"] == "10_20"


def test_predict_returns_float():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([14.58])

    mock_dv = MagicMock()
    mock_dv.transform.return_value = [[0, 1, 2.5]]

    features = {"PU_DO": "151_239", "trip_distance": 2.5}
    result = predict(mock_model, mock_dv, features)

    assert isinstance(result, float)
    assert result == 14.58


def test_predict_rounds_to_2_decimals():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([14.5812345])

    mock_dv = MagicMock()
    mock_dv.transform.return_value = [[0, 1, 2.5]]

    features = {"PU_DO": "151_239", "trip_distance": 2.5}
    result = predict(mock_model, mock_dv, features)

    assert result == 14.58
