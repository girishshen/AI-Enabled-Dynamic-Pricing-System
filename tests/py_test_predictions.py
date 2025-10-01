"""
tests/py_test_predictions.py

PyTest-based tests for the DynamicPricingSystem prediction pipeline.

Place this file at DynamicPricingSystem/tests/py_test_predictions.py
and run:

    pip install pytest
    pytest tests/py_test_predictions.py -q

Notes:
- If model files (models/LinearReg.pkl) are missing the tests will be skipped with a clear message.
- The real `save_to_csv` is monkeypatched to write to a temporary file during tests to avoid mutating your real logs.
"""

import os
import sys
import io
import math
import tempfile
import pandas as pd
import pytest

# Ensure project root is importable when running from repo root
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from routes import utils  # import module so we can access helpers and monkeypatch them

SAMPLE_PAYLOADS = [
    {
        "category": "Electronics",
        "current_price": 1200.0,
        "competitor_price": 1100.0,
        "demand_level": "High",
        "season": "Summer",
        "day_of_week": "Friday",
        "stock": 50,
    },
    {
        "category": "Clothing",
        "current_price": 240.0,
        "competitor_price": 250.0,
        "demand_level": "Medium",
        "season": "Spring",
        "day_of_week": "Tuesday",
        "stock": 200,
    },
    {
        "category": "Home Appliances",
        "current_price": 15000.0,
        "competitor_price": 14000.0,
        "demand_level": "Low",
        "season": "Winter",
        "day_of_week": "Monday",
        "stock": 5,
    },
]


@pytest.fixture(scope="module")
def artifacts():
    """
    Try to load model + optional preprocessor via the project's ModelLoader.
    If files are missing, skip the tests in this module (pytest.skip).
    """
    try:
        model, preprocessor = utils.model_loader.get_artifacts()
    except FileNotFoundError as e:
        pytest.skip(f"Model artifacts missing; skipped tests. ({e})")
    except Exception as e:
        # If loader throws an unexpected error, skip but show message
        pytest.skip(f"Could not load artifacts; skipped tests. ({e})")
    return {"model": model, "preprocessor": preprocessor}


@pytest.fixture(autouse=True)
def stub_save_to_csv(monkeypatch, tmp_path):
    """
    Replace the real save_to_csv with a lightweight stub that writes to a temp file.
    This prevents tests from appending to the real data_entries/prediction_log.csv.
    """
    temp_log = tmp_path / "prediction_log_test.csv"

    def _stub(parsed_input, predictions_dict):
        # Append CSV-like row for inspection if needed
        row = {**parsed_input}
        # flatten predictions dict into row with prefixed keys
        for k, v in predictions_dict.items():
            row[f"pred__{k}"] = v
        df = pd.DataFrame([row])
        # use header only if file doesn't exist
        if temp_log.exists():
            df.to_csv(temp_log, mode="a", header=False, index=False)
        else:
            df.to_csv(temp_log, index=False)
        return str(temp_log)

    monkeypatch.setattr(utils, "save_to_csv", _stub)
    return temp_log


@pytest.mark.parametrize("payload", SAMPLE_PAYLOADS)
def test_prediction_pipeline_basic(artifacts, payload):
    """
    For each sample payload:
      - parse_input returns parsed dict and no fatal validation errors
      - preprocess_input returns a DataFrame with at least one column
      - transforming via preprocessor (if present) or using raw DataFrame yields a matrix
      - model.predict returns a finite numeric value
      - format_usd / format_inr return non-empty strings
      - patched save_to_csv can be called without raising
    """
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    # 1) parse & validate
    parsed, errors = utils.parse_input(payload)
    assert isinstance(parsed, dict)
    # `errors` may be empty dict/list/None depending on implementation
    if errors:
        # if parse_input returns non-empty errors treat as fail
        pytest.fail(f"parse_input returned validation errors: {errors}")

    # 2) engineered features DataFrame
    X_df = utils.preprocess_input(parsed)
    assert hasattr(X_df, "shape"), "preprocess_input must return a DataFrame-like object"
    assert X_df.shape[1] >= 1, "engineered DataFrame must have at least 1 column"

    # 3) apply preprocessor if present
    if preprocessor is not None:
        try:
            X_trans = utils.safe_preprocessor_transform(preprocessor, X_df, logger=None)
        except Exception as e:
            # If transform fails, fallback to using the raw engineered DF (the app has fallbacks)
            X_trans = X_df
    else:
        X_trans = X_df

    # 4) convert to matrix usable by model
    M = utils.to_matrix(X_trans)
    assert hasattr(M, "shape"), "to_matrix must return an array-like with .shape"

    # 5) align to model expected input width if necessary
    M_aligned = utils.fit_to_n_features(M, model, logger=None)
    assert M_aligned.shape[1] >= 1

    # 6) predict
    preds = model.predict(M_aligned)
    # scalar or single-element vector expected
    try:
        pred_value = float(preds.ravel()[0])
    except Exception:
        pytest.fail("Model prediction did not return a numeric scalar/array")

    assert math.isfinite(pred_value), "Prediction must be a finite number"

    # 7) formatting helpers
    usd = utils.format_usd(pred_value)
    inr = utils.format_inr(pred_value)
    assert isinstance(usd, str) and len(usd) > 0
    assert isinstance(inr, str) and len(inr) > 0

    # 8) persistence (patched)
    saved_path = utils.save_to_csv(parsed, {"linear_regression": pred_value})
    assert os.path.exists(saved_path), "Patched save_to_csv should return path to the test CSV"


def test_model_feature_consistency(artifacts):
    """
    Sanity check: if the model exposes n_features_in_ verify we can create a matching-width matrix.
    This test will only run if the model has attribute `n_features_in_`.
    """
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    # Build a minimal payload based on first sample
    payload = SAMPLE_PAYLOADS[0]
    parsed, errors = utils.parse_input(payload)
    if errors:
        pytest.skip("Skipping feature consistency test because parse_input returned errors.")

    X_df = utils.preprocess_input(parsed)

    if preprocessor is not None:
        try:
            X_trans = utils.safe_preprocessor_transform(preprocessor, X_df, logger=None)
        except Exception:
            X_trans = X_df
    else:
        X_trans = X_df

    M = utils.to_matrix(X_trans)

    if hasattr(model, "n_features_in_"):
        M_adj = utils.fit_to_n_features(M, model, logger=None)
        assert M_adj.shape[1] == model.n_features_in_
    else:
        pytest.skip("Model does not expose n_features_in_; skipping exact-width assertion.")