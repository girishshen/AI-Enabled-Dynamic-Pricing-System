"""
tests/test_predictions.py

Standalone script to exercise the prediction pipeline in DynamicPricingSystem.

Place this file at DynamicPricingSystem/tests/test_predictions.py and run:
    python tests/test_predictions.py

It will:
 - Load Model + optional Preprocessor via model_loader from routes.utils
 - Create several sample payloads
 - Parse, preprocess, transform, predict, format and print the results
 - Append to data_entries/prediction_log.csv using save_to_csv (same helper used by the app)
"""

import pprint
import sys
import os

# Add project root to path if running from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from routes.utils import (
    model_loader,
    parse_input,
    preprocess_input,
    safe_preprocessor_transform,
    fit_to_n_features,
    to_matrix,
    format_usd,
    format_inr,
    save_to_csv,
)

pp = pprint.PrettyPrinter(indent=2)

SAMPLE_PAYLOADS = [
    # Typical electronics example
    {
        "category": "Electronics",
        "current_price": 1200.0,
        "competitor_price": 1100.0,
        "demand_level": "High",
        "season": "Summer",
        "day_of_week": "Friday",
        "stock": 50,
    },
    # Clothing, medium demand
    {
        "category": "Clothing",
        "current_price": 240.0,
        "competitor_price": 250.0,
        "demand_level": "Medium",
        "season": "Spring",
        "day_of_week": "Tuesday",
        "stock": 200,
    },
    # Low demand, home appliances, small stock
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


def run_single(payload: dict):
    logger = None
    try:
        # 1) parse & validate
        parsed, errors = parse_input(payload)
        if errors:
            print("Validation errors for payload:")
            pp.pprint(payload)
            pp.pprint(errors)
            return

        print("\n=== Parsed input ===")
        pp.pprint(parsed)

        # 2) get artifacts (linear model + optional preprocessor)
        linear_model, preprocessor = model_loader.get_artifacts()
        print("Loaded model:", type(linear_model).__name__)
        print("Preprocessor found:", preprocessor is not None)

        # 3) engineer features DataFrame
        X_df = preprocess_input(parsed)
        print("\nEngineered features (head):")
        try:
            print(X_df.head().to_markdown())
        except Exception:
            print(X_df.head())

        # 4) apply preprocessor if present
        if preprocessor is not None:
            try:
                X_trans = safe_preprocessor_transform(preprocessor, X_df, logger=None)
                M = to_matrix(X_trans)
            except Exception as e:
                print("Preprocessor.transform error:", e)
                # fallback: try aligning features (preprocessor missing / incompatible)
                M = to_matrix(X_df)
        else:
            M = to_matrix(X_df)

        # 5) ensure matrix width matches model expectations
        M = fit_to_n_features(M, linear_model, logger=None)

        # 6) predict
        try:
            preds = linear_model.predict(M)
        except Exception as e:
            print("Model prediction failed:", e)
            return

        pred_value = float(preds.ravel()[0])
        print("\nRaw prediction value:", pred_value)

        # 7) Format outputs (USD + INR helpers in utils)
        try:
            usd_str = format_usd(pred_value)
        except Exception:
            usd_str = f"${pred_value:,.2f}"
        try:
            inr_str = format_inr(pred_value)
        except Exception:
            inr_str = f"INR {pred_value:,.2f}"

        print("Formatted USD:", usd_str)
        print("Formatted INR:", inr_str)

        # 8) persist same as app
        save_to_csv(parsed, {"linear_regression": pred_value})
        print("Saved to data_entries/prediction_log.csv")

    except FileNotFoundError as fnf:
        print("Model file missing. Please ensure models/LinearReg.pkl is present.")
        print(fnf)
    except Exception as e:
        print("Unexpected error during test:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running prediction tests...")
    for p in SAMPLE_PAYLOADS:
        run_single(p)
    print("\nAll done.")