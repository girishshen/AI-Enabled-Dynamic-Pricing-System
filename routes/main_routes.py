from flask import Blueprint, render_template, request, jsonify, current_app
import traceback
import time

from .utils import (
    model_loader,
    parse_input,
    preprocess_input,
    save_to_csv,
    build_raw_df,
    align_features,
    to_matrix,
    fit_to_n_features,
    safe_preprocessor_transform,
    format_usd,
    format_inr,
    convert_usd_to_inr,
    number_to_words_intl,
    number_to_words_indian,
)


main_bp = Blueprint("main_bp", __name__)


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.post("/predict")
def predict():
    logger = current_app.logger
    t0 = time.perf_counter()

    try:
        payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
        parsed, errors = parse_input(payload or {})
        if errors:
            logger.warning("Validation errors: %s", errors)
            return (
                jsonify({
                    "success": False,
                    "errors": errors,
                    "message": "Validation failed. Please correct the inputs.",
                }),
                400,
            )

        # Ensure no None values in required numerics
        if parsed["current_price"] is None or parsed["competitor_price"] is None or parsed["stock"] is None:
            return (
                jsonify({
                    "success": False,
                    "errors": ["Invalid numeric inputs"],
                }),
                400,
            )

        # Load artifacts
        lin_model, preprocessor = model_loader.get_artifacts()

        # Build inputs
        X_for_linear = None
        if preprocessor is not None:
            pre_df = build_raw_df(parsed)
            try:
                Xp = safe_preprocessor_transform(preprocessor, pre_df, logger)
                X_for_linear = to_matrix(Xp)
            except Exception as e:
                logger.warning("Preprocessor.transform failed, falling back to engineered features: %s", e)

        if X_for_linear is None:
            X_eng = preprocess_input(parsed)
            X_aligned = align_features(X_eng, lin_model)
            X_for_linear = to_matrix(X_aligned)

        X_for_linear = fit_to_n_features(X_for_linear, lin_model, logger)

        # Base prediction is in USD numeric
        base_usd = float(lin_model.predict(X_for_linear)[0])

        # Determine display mode
        mode = (parsed.get("display") or "usd").lower()
        if mode == "usd":
            display_value = format_usd(base_usd)
            currency = "USD"
            numeric = round(base_usd, 2)
        elif mode == "inr":
            inr_val = convert_usd_to_inr(base_usd)
            display_value = format_inr(inr_val)
            currency = "INR"
            numeric = round(inr_val, 2)
        elif mode == "words-intl":
            display_value = number_to_words_intl(base_usd)
            currency = "USD"
            numeric = round(base_usd, 2)
        elif mode == "words-indian":
            inr_val = convert_usd_to_inr(base_usd)
            display_value = number_to_words_indian(inr_val)
            currency = "INR"
            numeric = round(inr_val, 2)
        else:
            # default to USD
            display_value = format_usd(base_usd)
            currency = "USD"
            numeric = round(base_usd, 2)

        predictions = {
            "linear_regression": round(base_usd, 2),  # always return base USD numeric
        }

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Log
        logger.info(
            "Predictions made | input=%s | base_usd=%.2f | mode=%s | display=%s | duration_ms=%.1f",
            parsed,
            base_usd,
            mode,
            display_value,
            duration_ms,
        )

        # Persist to CSV (store base USD and chosen display)
        save_to_csv(parsed, {**predictions, "display_value": display_value, "display_currency": currency, "duration_ms": duration_ms})

        return jsonify({
            "success": True,
            "predictions": predictions,
            "display": {
                "mode": mode,
                "currency": currency,
                "value": display_value,
                "numeric": numeric,
            },
            "duration_ms": duration_ms,
        })

    except FileNotFoundError as fnf:
        logger.error("Model file missing: %s", str(fnf))
        return jsonify({
            "success": False,
            "message": "Required model files missing. Ensure LinearReg.pkl exists; optional preprocessor.pkl can also be placed in the models directory.",
            "errors": [str(fnf)],
        }), 500

    except Exception as e:
        logger.error("Prediction error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Prediction failed due to an internal error.",
            "errors": [str(e)],
        }), 500