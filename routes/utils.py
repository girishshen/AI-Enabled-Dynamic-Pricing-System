import os
import threading
import logging
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_entries")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

FILE_LOCK = threading.Lock()

# ===== Currency & Formatting Config =====
USD_TO_INR_RATE = 83.0  # Update if you need a different conversion rate


def format_usd(n: float) -> str:
    return f"${n:,.2f}"


def _format_indian_grouping(n: float) -> str:
    # Indian grouping: last 3 digits, then groups of 2
    s = f"{abs(n):.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        pre = whole[:-3]
        last3 = whole[-3:]
        parts = []
        while len(pre) > 2:
            parts.insert(0, pre[-2:])
            pre = pre[:-2]
        if pre:
            parts.insert(0, pre)
        grouped = ",".join(parts) + "," + last3
    else:
        grouped = whole
    sign = "-" if n < 0 else ""
    return f"{sign}₹{grouped}.{frac}"


def format_inr(n: float) -> str:
    return _format_indian_grouping(n)


def convert_usd_to_inr(usd: float, rate: float = USD_TO_INR_RATE) -> float:
    return float(usd) * float(rate)


# ===== Words Formatting =====

def number_to_words_intl(num: float) -> str:
    a = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    b = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    def in_words(n: int) -> str:
        if n < 20:
            return a[n]
        if n < 100:
            return b[n // 10] + ("-" + a[n % 10] if n % 10 else "")
        if n < 1000:
            return a[n // 100] + " hundred" + (" " + in_words(n % 100) if n % 100 else "")
        if n < 1_000_000:
            return in_words(n // 1_000) + " thousand" + (" " + in_words(n % 1_000) if n % 1_000 else "")
        if n < 1_000_000_000:
            return in_words(n // 1_000_000) + " million" + (" " + in_words(n % 1_000_000) if n % 1_000_000 else "")
        if n < 1_000_000_000_000:
            return in_words(n // 1_000_000_000) + " billion" + (" " + in_words(n % 1_000_000_000) if n % 1_000_000_000 else "")
        return str(n)

    whole = int(abs(num))
    frac = int(round((abs(num) - whole) * 100))
    sign = "negative " if num < 0 else ""
    words = in_words(whole) or "zero"
    return f"{sign}{words}{' and ' + str(frac).zfill(2) + '/100' if frac else ''}"


def number_to_words_indian(num: float) -> str:
    a = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    b = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    def two(n: int) -> str:
        if n < 20:
            return a[n]
        return b[n // 10] + ("-" + a[n % 10] if n % 10 else "")

    def three(n: int) -> str:
        if n < 100:
            return two(n)
        return a[n // 100] + " hundred" + (" " + two(n % 100) if n % 100 else "")

    whole = int(abs(num))
    frac = int(round((abs(num) - whole) * 100))
    sign = "negative " if num < 0 else ""

    crore = whole // 10_000_000
    lakh = (whole % 10_000_000) // 100_000
    thousand = (whole % 100_000) // 1_000
    hundred = whole % 1_000

    parts = []
    if crore:
        parts.append(three(crore) + " crore")
    if lakh:
        parts.append(three(lakh) + " lakh")
    if thousand:
        parts.append(three(thousand) + " thousand")
    if hundred:
        parts.append(three(hundred))

    words = " ".join(parts) if parts else "zero"
    return f"{sign}{words}{' and ' + str(frac).zfill(2) + '/100' if frac else ''}"


class ModelLoader:
    """Thread-safe, lazy artifact loader for linear model + optional preprocessor."""

    _lock = threading.Lock()
    _linear_model = None
    _preprocessor = None

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def _load_linear(self) -> None:
        if self._linear_model is None:
            lin_path = os.path.join(MODELS_DIR, "LinearReg.pkl")
            if not os.path.exists(lin_path):
                self.logger.error("LinearReg.pkl not found at %s", lin_path)
                raise FileNotFoundError(f"Missing model file: {lin_path}")
            self._linear_model = joblib.load(lin_path)
            self.logger.info("Linear Regression model loaded")

    def _load_preprocessor(self) -> None:
        # Optional preprocessor
        if self._preprocessor is None:
            pp_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
            if os.path.exists(pp_path):
                try:
                    self._preprocessor = joblib.load(pp_path)
                    self.logger.info("Preprocessor loaded")
                except Exception as e:
                    self.logger.warning("Failed to load preprocessor.pkl: %s", e)
                    self._preprocessor = None
            else:
                self._preprocessor = None

    def get_artifacts(self) -> Tuple[Any, Optional[Any]]:
        # Double-checked locking pattern
        if self._linear_model is None or self._preprocessor is None:
            with self._lock:
                if self._linear_model is None:
                    self._load_linear()
                if self._preprocessor is None:
                    self._load_preprocessor()
        return self._linear_model, self._preprocessor


# Shared singleton instance
model_loader = ModelLoader()


ALLOWED_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home Appliances",
    "Beauty",
    "Books ",
    "Sports",
    "Toys",
]

ALLOWED_SEASONS = [
    "Spring",
    "Summer",
    "Autumn",
    "Festival",
    "Monsoon",
    "Winter",
]

ALLOWED_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

ALLOWED_DEMAND = ["Low", "Medium", "High"]

DEMAND_INDEX = {"Low": 1.0, "Medium": 2.0, "High": 3.0}

# ========== Validation & Parsing ==========

def parse_input(data: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    """Validate and parse incoming request data into proper types."""
    errors: list[str] = []

    category = data.get("category", "Electronics")
    if category not in ALLOWED_CATEGORIES:
        errors.append("Invalid category")

    demand_level = data.get("demand_level", "Medium")
    if demand_level not in ALLOWED_DEMAND:
        errors.append("Invalid demand level")

    season = data.get("season", "Summer")
    if season not in ALLOWED_SEASONS:
        errors.append("Invalid season")

    day_of_week = data.get("day_of_week", "Monday")
    if day_of_week not in ALLOWED_DAYS:
        errors.append("Invalid day of week")

    # Numeric parsing
    def parse_float(key: str, min_value: float | None = None) -> float | None:
        val = data.get(key)
        try:
            f = float(val)
            if min_value is not None and f < min_value:
                errors.append(f"{key.replace('_', ' ').title()} must be ≥ {min_value}")
            return f
        except (TypeError, ValueError):
            errors.append(f"Invalid {key.replace('_', ' ')}")
            return None

    def parse_int(key: str, min_value: int | None = None) -> int | None:
        val = data.get(key)
        try:
            i = int(float(val))
            if min_value is not None and i < min_value:
                errors.append(f"{key.replace('_', ' ').title()} must be ≥ {min_value}")
            return i
        except (TypeError, ValueError):
            errors.append(f"Invalid {key.replace('_', ' ')}")
            return None

    current_price = parse_float("current_price", min_value=0.0)
    competitor_price = parse_float("competitor_price", min_value=0.0)
    stock = parse_int("stock", min_value=0)

    display = (data.get("display") or "usd").lower()

    parsed = {
        "category": category,
        "current_price": current_price,
        "competitor_price": competitor_price,
        "demand_level": demand_level,
        "season": season,
        "day_of_week": day_of_week,
        "stock": stock,
        "display": display,
    }

    return parsed, errors


# ========== Feature Builders ==========

def _ohe(value: str, allowed: list[str], prefix: str) -> Dict[str, int]:
    return {f"{prefix}_{opt}": 1 if value == opt else 0 for opt in allowed}


def build_engineered(parsed: Dict[str, Any]) -> Dict[str, Any]:
    cp = float(parsed["current_price"])
    cpp = float(parsed["competitor_price"])
    demand_idx = DEMAND_INDEX.get(parsed["demand_level"], 1.0)

    price_diff = cp - cpp
    price_ratio = cp / (cpp + 1e-9)
    price_demand_inter = cp * demand_idx
    log_price = np.log1p(cp)
    log_comp_price = np.log1p(cpp)

    return {
        "Price_Diff": price_diff,
        "Price_Ratio": price_ratio,
        "Price_Demand_Interaction": price_demand_inter,
        "Log_Current_Price": log_price,
        "Log_Competitor_Price": log_comp_price,
        "Cost_per_Unit": 0.0,
    }


def build_raw_df(parsed: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "Category": parsed["category"],
        "Season": parsed["season"],
        "Day_of_Week": parsed["day_of_week"],
        "Demand_Level": parsed["demand_level"],
        "Current_Price": float(parsed["current_price"]),
        "Competitor_Price": float(parsed["competitor_price"]),
        "Stock": int(parsed["stock"]) if parsed["stock"] is not None else 0,
    }
    row.update(build_engineered(parsed))
    return pd.DataFrame([row])


def preprocess_input(parsed: Dict[str, Any]) -> pd.DataFrame:
    stock = int(parsed["stock"]) if parsed["stock"] is not None else 0

    features: Dict[str, Any] = {
        "Current_Price": float(parsed["current_price"]),
        "Competitor_Price": float(parsed["competitor_price"]),
        "Stock": stock,
        "Category": parsed["category"],
        "Season": parsed["season"],
        "Day_of_Week": parsed["day_of_week"],
        "Demand_Level": parsed["demand_level"],
        **build_engineered(parsed),
        "demand_index": DEMAND_INDEX.get(parsed["demand_level"], 1.0),
    }

    ohe = {}
    ohe.update(_ohe(parsed["category"], ALLOWED_CATEGORIES, "cat"))
    ohe.update(_ohe(parsed["season"], ALLOWED_SEASONS, "season"))
    ohe.update(_ohe(parsed["day_of_week"], ALLOWED_DAYS, "dow"))
    features.update(ohe)

    ordered = {k: features[k] for k in sorted(features.keys())}
    return pd.DataFrame([ordered])


# ========== Utilities for alignment ==========

def align_features(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)
    try:
        if names is not None:
            return X.reindex(columns=list(names), fill_value=0)
    except Exception:
        pass
    return X


def to_matrix(X: Any) -> np.ndarray:
    try:
        if hasattr(X, "values"):
            X = X.values
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X)
    except Exception:
        return np.asarray(X)


def fit_to_n_features(M: np.ndarray, model: Any, logger: Optional[logging.Logger] = None) -> np.ndarray:
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            expected = coef.shape[-1]
    if expected is None:
        return M
    n = M.shape[1]
    if n == expected:
        return M
    if n > expected:
        if logger:
            logger.warning("Input has %s features but model expects %s; truncating.", n, expected)
        return M[:, :expected]
    if logger:
        logger.warning("Input has %s features but model expects %s; padding with zeros.", n, expected)
    pad = np.zeros((M.shape[0], expected - n), dtype=M.dtype)
    return np.hstack([M, pad])


def safe_preprocessor_transform(preprocessor: Any, df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Any:
    try:
        return preprocessor.transform(df)
    except Exception as e:
        msg = str(e)
        missing = []
        if "columns are missing" in msg and "{" in msg and "}" in msg:
            try:
                inside = msg.split("{", 1)[1].rsplit("}", 1)[0]
                for token in inside.split(','):
                    col = token.strip().strip("'\"")
                    if col:
                        missing.append(col)
            except Exception:
                pass
        if missing:
            for col in missing:
                if col not in df.columns:
                    df[col] = 0
            if logger:
                logger.warning("Retrying preprocessor.transform after adding missing columns: %s", missing)
            return preprocessor.transform(df)
        raise

# ========== Persistence ==========

def save_to_csv(input_data: Dict[str, Any], predictions: Dict[str, float]) -> str:
    """Append the combined input and prediction row to a CSV file safely.

    Writes/creates data_entries/prediction_log.csv with a header on first write.
    Thread-safe via FILE_LOCK.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        **input_data,
        **predictions,
    }
    out_path = os.path.join(DATA_DIR, "prediction_log.csv")
    df = pd.DataFrame([row])

    with FILE_LOCK:
        file_exists = os.path.exists(out_path)
        df.to_csv(out_path, mode="a", header=not file_exists, index=False)

    return out_path