"""
agents/state.py
---------------
Shared state passed between all agents in the LangGraph pipeline.

Design principle: every field is either an INPUT (set once at start),
or set by exactly ONE agent. No agent overwrites another agent's outputs.
This makes the audit trail clear and reproducible.
"""

from typing import TypedDict, Optional
from datetime import datetime


class AgentStep(TypedDict):
    """One entry in the audit trail."""
    timestamp: str           # ISO format
    agent: str               # 'underwriter' | 'compliance' | 'explainer'
    action: str              # short description
    output_summary: str      # first 200 chars of agent's contribution
    tools_called: list[str]  # which tools this agent used


class ShapFeature(TypedDict):
    """One SHAP feature contribution."""
    name: str                # feature name as in model
    value: float             # actual feature value for this borrower
    contribution: float      # SHAP value (signed)
    direction: str           # 'increases_pd' | 'decreases_pd'
    plain_language: Optional[str]  # filled by explainer agent


class CreditDecisionState(TypedDict):
    """
    The shared state object that flows through the LangGraph pipeline.
    """
    # ─── Input (set once at START) ──────────────────────────────────
    application_id: str
    application: dict                       # raw borrower features
    
    # ─── Underwriter agent outputs ──────────────────────────────────
    pd_score: Optional[float]               # calibrated probability of default
    decision_band: Optional[str]            # 'approve' | 'review' | 'decline'
    underwriter_reasoning: Optional[str]    # 3-sentence justification
    shap_top_features: Optional[list[ShapFeature]]
    
    # ─── Compliance agent outputs ───────────────────────────────────
    compliance_status: Optional[str]        # 'pass' | 'flagged' | 'fail'
    compliance_citations: Optional[list[str]]   # specific regulatory clauses
    required_disclosures: Optional[list[str]]   # things AAN must include
    protected_class_concern: Optional[bool]     # True if proxy detected
    
    # ─── Explainer agent outputs ────────────────────────────────────
    customer_letter_en: Optional[str]
    customer_letter_sv: Optional[str]       # Swedish version
    
    # ─── Final routing ──────────────────────────────────────────────
    final_decision: Optional[str]           # APPROVE | HUMAN_REVIEW | DECLINE
    
    # ─── Audit trail ────────────────────────────────────────────────
    audit_trail: list[AgentStep]            # append-only log


def make_initial_state(application_id: str, application: dict) -> CreditDecisionState:
    """Create a fresh state with just the input fields populated."""
    return CreditDecisionState(
        application_id=application_id,
        application=application,
        pd_score=None,
        decision_band=None,
        underwriter_reasoning=None,
        shap_top_features=None,
        compliance_status=None,
        compliance_citations=None,
        required_disclosures=None,
        protected_class_concern=None,
        customer_letter_en=None,
        customer_letter_sv=None,
        final_decision=None,
        audit_trail=[],
    )


def append_audit_step(
    state: CreditDecisionState,
    agent: str,
    action: str,
    output_summary: str,
    tools_called: list[str],
) -> list[AgentStep]:
    """Append one step to the audit trail. Returns the new list."""
    step = AgentStep(
        timestamp=datetime.utcnow().isoformat(),
        agent=agent,
        action=action,
        output_summary=output_summary[:200],
        tools_called=tools_called,
    )
    return state['audit_trail'] + [step]
