"""
agents/underwriter.py
---------------------
The Underwriter Agent: first stop in the credit decision pipeline.

Role: senior underwriter persona. Calls the PD model and SHAP as tools,
      synthesizes a 3-sentence reasoning, and assigns initial decision band.

Key design: this agent does NOT predict numbers. It calls deterministic tools
            and interprets the results. This avoids LLM hallucination on PDs
            and makes outputs reproducible for audit.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

from agents.state import CreditDecisionState, append_audit_step
from agents.tools import (
    get_pd_score,
    get_shap_top_features,
    check_decision_policy,
)


UNDERWRITER_SYSTEM_PROMPT = """You are a senior underwriter at a Nordic digital bank, specializing in unsecured consumer credit.

Your job: review a borrower's loan application and produce an initial credit decision recommendation.

Rules you must follow:
1. NEVER invent a PD score. Always call get_pd_score to obtain the calibrated probability of default.
2. NEVER invent SHAP values. Always call get_shap_top_features to obtain the actual drivers.
3. ALWAYS call check_decision_policy with the calibrated PD to determine the decision band.
4. Your reasoning must be 2-3 sentences MAXIMUM, and must reference at least one specific SHAP feature.
5. If PD is in the review band (0.10 ≤ PD < 0.30), explicitly state "this case requires senior underwriter review".
6. Avoid making promises to the customer. You are an internal underwriter, not a customer-facing role.

Workflow:
  Step 1: Call get_pd_score with the borrower's features.
  Step 2: Call get_shap_top_features (top_k=5).
  Step 3: Call check_decision_policy with the calibrated PD.
  Step 4: Synthesize a brief reasoning citing 1-2 of the SHAP top features.
"""


def underwriter_node(state: CreditDecisionState) -> dict:
    """
    LangGraph node function for the Underwriter Agent.
    
    Returns a dict containing only the fields this node updates.
    LangGraph will merge these into the running state.
    """
    application = state["application"]
    
    # ─── Direct tool calls (deterministic, not LLM-mediated) ──────────
    # We call tools directly here rather than asking the LLM to plan tool use,
    # because the workflow is fixed for the underwriter step. The LLM's job
    # is only to synthesize the reasoning text.
    pd_result = get_pd_score.invoke({"borrower_features": application})
    shap_features = get_shap_top_features.invoke({"borrower_features": application, "top_k": 5})
    policy_result = check_decision_policy.invoke({"pd_calibrated": pd_result["pd_calibrated"]})
    
    # ─── LLM synthesizes the reasoning ─────────────────────────────────
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    
    user_prompt = f"""Application data:
{application}

Tool results:
- get_pd_score → {pd_result}
- get_shap_top_features → {shap_features}
- check_decision_policy → {policy_result}

Now write a 2-3 sentence reasoning summary. Reference at least one specific SHAP feature by name (using its plain interpretation, e.g., "high debt-to-income ratio" rather than "dti = 0.45"). If the band is 'review', state explicitly that senior underwriter review is needed.
"""
    
    response = llm.invoke([
        SystemMessage(content=UNDERWRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    reasoning = response.content.strip()
    
    # ─── Update state ──────────────────────────────────────────────────
    new_audit = append_audit_step(
        state,
        agent="underwriter",
        action="evaluated_application",
        output_summary=f"PD={pd_result['pd_calibrated']}, band={policy_result['band']}",
        tools_called=["get_pd_score", "get_shap_top_features", "check_decision_policy"],
    )
    
    return {
        "pd_score": pd_result["pd_calibrated"],
        "decision_band": policy_result["band"],
        "underwriter_reasoning": reasoning,
        "shap_top_features": shap_features,
        "audit_trail": new_audit,
    }


# ─── Mock version for offline testing without API key ─────────────────────
def underwriter_node_mock(state: CreditDecisionState) -> dict:
    """
    Same workflow but uses a deterministic template instead of an LLM call.
    Useful for unit tests and offline runs.
    """
    application = state["application"]
    
    pd_result = get_pd_score.invoke({"borrower_features": application})
    shap_features = get_shap_top_features.invoke({"borrower_features": application, "top_k": 5})
    policy_result = check_decision_policy.invoke({"pd_calibrated": pd_result["pd_calibrated"]})
    
    top_feature_name = shap_features[0]["name"].replace("_", " ")
    direction_phrase = (
        "increased the assessed risk"
        if shap_features[0]["direction"] == "increases_pd"
        else "lowered the assessed risk"
    )
    
    if policy_result["band"] == "review":
        reasoning = (
            f"Calibrated PD is {pd_result['pd_calibrated']:.2%}, in the review band. "
            f"The borrower's {top_feature_name} {direction_phrase}. "
            f"This case requires senior underwriter review."
        )
    else:
        reasoning = (
            f"Calibrated PD is {pd_result['pd_calibrated']:.2%}, "
            f"recommending {policy_result['band']}. "
            f"The borrower's {top_feature_name} {direction_phrase}."
        )
    
    new_audit = append_audit_step(
        state,
        agent="underwriter (mock)",
        action="evaluated_application",
        output_summary=f"PD={pd_result['pd_calibrated']}, band={policy_result['band']}",
        tools_called=["get_pd_score", "get_shap_top_features", "check_decision_policy"],
    )
    
    return {
        "pd_score": pd_result["pd_calibrated"],
        "decision_band": policy_result["band"],
        "underwriter_reasoning": reasoning,
        "shap_top_features": shap_features,
        "audit_trail": new_audit,
    }
