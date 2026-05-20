"""
agents/tools.py
---------------
Tool wrappers around the existing llm-credit-risk PD model and SHAP explainer.

Key design principle: agents do NOT predict PD or compute SHAP themselves.
They CALL these tools. The model is the deterministic source of truth;
the agent's job is to orchestrate, interpret, and route.

This separation is important for:
  - Reproducibility (model output is deterministic)
  - Auditability (we know exactly what the model said)
  - Avoiding LLM hallucination on numeric values
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from langchain_core.tools import tool


# ─── Paths to existing artifacts (from llm-credit-risk repo) ──────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent / "llm-credit-risk"
MODEL_PATH = REPO_ROOT / "outputs" / "model_C.json"
CALIBRATOR_PATH = REPO_ROOT / "outputs" / "calibrator_C.pkl"
SHAP_SAMPLE_PATH = REPO_ROOT / "outputs" / "shap_sample.csv"
FEATURE_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "data" / "compliance_docs" / "feature_glossary.csv"


# ─── Lazy-loaded singletons ───────────────────────────────────────────────
_model = None
_calibrator = None
_glossary = None


def _load_model():
    """Load the XGBoost challenger model from existing repo."""
    global _model
    if _model is None:
        _model = xgb.Booster()
        _model.load_model(str(MODEL_PATH))
    return _model


def _load_calibrator():
    """Load the isotonic / sigmoid calibrator."""
    global _calibrator
    if _calibrator is None:
        with open(CALIBRATOR_PATH, "rb") as f:
            _calibrator = pickle.load(f)
    return _calibrator


def _load_glossary() -> dict:
    """Feature name → plain language mapping."""
    global _glossary
    if _glossary is None:
        df = pd.read_csv(FEATURE_GLOSSARY_PATH)
        _glossary = dict(zip(df["feature"], df["plain_language"]))
    return _glossary


# ─── Tool 1: get calibrated PD score ──────────────────────────────────────
@tool
def get_pd_score(borrower_features: dict) -> dict:
    """
    Compute the calibrated probability of default (PD) for a borrower.
    
    Args:
        borrower_features: dict mapping feature name to value, matching the
                           feature schema of model_C.
    
    Returns:
        dict with keys:
          - pd_calibrated: float in [0, 1], the calibrated PD
          - pd_raw: float, raw model output before calibration
          - model_version: str, identifies which model was used
    """
    model = _load_model()
    calibrator = _load_calibrator()
    
    df = pd.DataFrame([borrower_features])
    dmatrix = xgb.DMatrix(df)
    pd_raw = float(model.predict(dmatrix)[0])
    pd_calibrated = float(calibrator.transform([pd_raw])[0])
    
    return {
        "pd_calibrated": round(pd_calibrated, 4),
        "pd_raw": round(pd_raw, 4),
        "model_version": "model_C_v1",
    }


# ─── Tool 2: get SHAP top features ────────────────────────────────────────
@tool
def get_shap_top_features(borrower_features: dict, top_k: int = 5) -> list[dict]:
    """
    Get the top-k features driving this borrower's PD prediction, ranked by
    absolute SHAP value.
    
    Args:
        borrower_features: dict matching model schema
        top_k: how many top features to return (default 5)
    
    Returns:
        list of dicts with name, value, contribution, direction.
        Sorted by abs(contribution) descending.
    """
    import shap
    
    model = _load_model()
    df = pd.DataFrame([borrower_features])
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)[0]
    
    feature_names = list(df.columns)
    feature_values = df.iloc[0].to_dict()
    
    contributions = []
    for name, contrib in zip(feature_names, shap_values):
        contributions.append({
            "name": name,
            "value": feature_values[name],
            "contribution": round(float(contrib), 4),
            "direction": "increases_pd" if contrib > 0 else "decreases_pd",
        })
    
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_k]


# ─── Tool 3: decision band policy ─────────────────────────────────────────
@tool
def check_decision_policy(pd_calibrated: float) -> dict:
    """
    Apply the bank's risk appetite policy to convert PD into a decision band.
    
    Thresholds reflect a typical consumer credit risk appetite:
      - PD < 0.10: low risk, recommend approve
      - 0.10 ≤ PD < 0.30: marginal, recommend manual review
      - PD ≥ 0.30: high risk, recommend decline
    
    Args:
        pd_calibrated: calibrated PD from get_pd_score
    
    Returns:
        dict with band, threshold_used, and brief rationale.
    """
    if pd_calibrated < 0.10:
        return {
            "band": "approve",
            "threshold_used": 0.10,
            "rationale": "PD below 10% threshold; within standard risk appetite.",
        }
    elif pd_calibrated < 0.30:
        return {
            "band": "review",
            "threshold_used": 0.30,
            "rationale": "PD in 10-30% range; marginal case requiring senior underwriter review.",
        }
    else:
        return {
            "band": "decline",
            "threshold_used": 0.30,
            "rationale": "PD exceeds 30% threshold; outside standard risk appetite.",
        }


# ─── Tool 4: protected-class proxy detection ──────────────────────────────
# Used by Compliance Agent
PROTECTED_CLASS_PROXIES = {
    "zip_code", "zipcode", "postal_code",  # geographic proxy for ethnicity
    "marital_status", "married",            # marital status (Reg B protected)
    "age", "age_group",                     # age-based decisions need extra scrutiny
}


@tool
def check_protected_class_features(top_features: list[dict]) -> dict:
    """
    Check if any top-driving SHAP features are proxies for protected classes
    under ECOA Reg B.
    
    Args:
        top_features: output of get_shap_top_features
    
    Returns:
        dict with concern flag, list of suspicious features, and rationale.
    """
    suspicious = []
    for feat in top_features:
        name_lower = feat["name"].lower()
        for proxy in PROTECTED_CLASS_PROXIES:
            if proxy in name_lower:
                suspicious.append(feat["name"])
                break
    
    return {
        "concern": len(suspicious) > 0,
        "suspicious_features": suspicious,
        "rationale": (
            f"Found {len(suspicious)} feature(s) that may proxy for "
            f"protected classes under Reg B: {suspicious}."
            if suspicious
            else "No top features appear to proxy for protected classes."
        ),
    }


# ─── Tool 5: translate SHAP feature to plain language ─────────────────────
@tool
def translate_feature_to_plain(feature_name: str, value: float, direction: str) -> str:
    """
    Convert a SHAP feature into customer-facing plain language for AAN.
    
    Args:
        feature_name: technical feature name (e.g. 'dti')
        value: actual feature value for this borrower
        direction: 'increases_pd' or 'decreases_pd'
    
    Returns:
        plain-language sentence suitable for an Adverse Action Notice.
    """
    glossary = _load_glossary()
    template = glossary.get(feature_name)
    
    if template is None:
        # Fallback: generic phrasing
        return f"Your {feature_name.replace('_', ' ')} value of {value} affected the decision."
    
    return template.format(value=value)


# ─── List of all tools, for binding to LLM ────────────────────────────────
ALL_TOOLS = [
    get_pd_score,
    get_shap_top_features,
    check_decision_policy,
    check_protected_class_features,
    translate_feature_to_plain,
]
