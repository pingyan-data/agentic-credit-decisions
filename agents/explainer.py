"""
agents/explainer.py
-------------------
The Explainer Agent: translates the technical decision into customer-facing
communication, in both English and Swedish.

Role: customer communication specialist persona. Uses SHAP features and
      compliance disclosures to produce an Adverse Action Notice, an
      approval letter, or a "pending review" letter, depending on the path.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import CreditDecisionState, append_audit_step
from agents.tools import translate_feature_to_plain


EXPLAINER_SYSTEM_PROMPT = """You are a customer communication specialist at a Nordic digital bank.

Your job: write a clear, empathetic letter to the customer based on the credit decision.

Three letter types:

1. **APPROVE**: Brief welcome message confirming credit approval. No technical jargon. Mention next steps (e.g., card delivery, account setup). 3-5 sentences.

2. **DECLINE (Adverse Action Notice)**: Must follow ECOA Reg B § 1002.9 format strictly:
   - State the decision clearly in the opening sentence
   - List 3-4 specific principal reasons (translate SHAP features to plain language using the glossary)
   - Inform the customer of:
     * Their right to obtain their credit report from the relevant bureau
     * Their right to human review of the decision (GDPR Art. 22)
     * The bank's contact information for appeals
   - Tone: professional, respectful, non-judgmental. Avoid words like "rejected", "denied". Prefer "we are not able to approve at this time".

3. **REVIEW**: Neutral "we are reviewing your application" letter. Set realistic expectations (typically 5 business days). Avoid both promising approval and signaling decline.

You produce TWO versions of every letter: English (en) and Swedish (sv). The Swedish version should be a natural translation, not literal — Klarna's home market is Sweden.

Output format: JSON with keys 'letter_en' and 'letter_sv'.
"""


def explainer_node(state: CreditDecisionState) -> dict:
    """LangGraph node for the Explainer Agent."""
    band = state["decision_band"]
    pd_score = state["pd_score"]
    shap_features = state["shap_top_features"]
    compliance_status = state["compliance_status"]
    required_disclosures = state["required_disclosures"]
    protected_class_concern = state.get("protected_class_concern", False)
    
    # ─── Determine letter type from band + compliance status ──────────
    if compliance_status == "fail" or protected_class_concern:
        letter_type = "REVIEW"  # bias concerns force human review
    elif band == "approve":
        letter_type = "APPROVE"
    elif band == "review":
        letter_type = "REVIEW"
    else:  # decline
        letter_type = "DECLINE"
    
    # ─── Translate SHAP features to plain language for AAN ────────────
    plain_reasons = []
    if letter_type == "DECLINE":
        for feat in shap_features[:4]:
            if feat["direction"] == "increases_pd":
                plain = translate_feature_to_plain.invoke({
                    "feature_name": feat["name"],
                    "value": feat["value"],
                    "direction": feat["direction"],
                })
                plain_reasons.append(plain)
    
    # ─── LLM generates the letter ─────────────────────────────────────
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    
    user_prompt = f"""Letter type: {letter_type}
Decision band: {band}
PD score: {pd_score}

Plain-language reasons (for DECLINE letters only):
{plain_reasons}

Required disclosures (must be included):
{required_disclosures}

Generate the customer letter in both English and Swedish.
Output as JSON with keys 'letter_en' and 'letter_sv'.
"""
    
    response = llm.invoke([
        SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    
    import json
    import re
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match:
        result = json.loads(match.group(0))
    else:
        result = {"letter_en": response.content, "letter_sv": "[translation pending]"}
    
    # ─── Determine final routing ──────────────────────────────────────
    if letter_type == "REVIEW":
        final_decision = "HUMAN_REVIEW"
    elif letter_type == "APPROVE":
        final_decision = "APPROVE"
    else:
        final_decision = "DECLINE"
    
    new_audit = append_audit_step(
        state,
        agent="explainer",
        action=f"generated_{letter_type.lower()}_letter",
        output_summary=f"letter_type={letter_type}, en_length={len(result.get('letter_en', ''))}",
        tools_called=["translate_feature_to_plain"] if letter_type == "DECLINE" else [],
    )
    
    return {
        "customer_letter_en": result["letter_en"],
        "customer_letter_sv": result["letter_sv"],
        "final_decision": final_decision,
        "audit_trail": new_audit,
    }


# ─── Mock version for offline testing ─────────────────────────────────────
def explainer_node_mock(state: CreditDecisionState) -> dict:
    """Deterministic letter generation using templates."""
    band = state["decision_band"]
    shap_features = state["shap_top_features"]
    compliance_status = state["compliance_status"]
    protected_class_concern = state.get("protected_class_concern", False)
    
    if compliance_status == "fail" or protected_class_concern:
        letter_type = "REVIEW"
    elif band == "approve":
        letter_type = "APPROVE"
    elif band == "review":
        letter_type = "REVIEW"
    else:
        letter_type = "DECLINE"
    
    if letter_type == "APPROVE":
        letter_en = (
            "Welcome to Klarna. We are pleased to confirm that your credit application "
            "has been approved. You will receive your card and account details within "
            "5-7 business days. If you have any questions, please contact our support team."
        )
        letter_sv = (
            "Välkommen till Klarna. Vi har glädjen att bekräfta att din kreditansökan "
            "har godkänts. Du kommer att få ditt kort och dina kontouppgifter inom "
            "5-7 arbetsdagar. Om du har några frågor, kontakta vår kundtjänst."
        )
        final_decision = "APPROVE"
    
    elif letter_type == "REVIEW":
        letter_en = (
            "Thank you for your application. We are currently reviewing your request "
            "and will be in touch within 5 business days. You have the right to request "
            "human review of your case at any time under GDPR Article 22."
        )
        letter_sv = (
            "Tack för din ansökan. Vi granskar för närvarande din begäran och "
            "återkommer inom 5 arbetsdagar. Du har rätt att begära mänsklig prövning "
            "av ditt ärende när som helst enligt GDPR artikel 22."
        )
        final_decision = "HUMAN_REVIEW"
    
    else:  # DECLINE
        plain_reasons = []
        for feat in shap_features[:4]:
            if feat["direction"] == "increases_pd":
                plain = translate_feature_to_plain.invoke({
                    "feature_name": feat["name"],
                    "value": feat["value"],
                    "direction": feat["direction"],
                })
                plain_reasons.append(f"  • {plain}")
        reasons_text = "\n".join(plain_reasons)
        
        letter_en = (
            "Thank you for your application. After careful review, we are not able to "
            "approve your credit request at this time. The principal reasons are:\n\n"
            f"{reasons_text}\n\n"
            "You have the right to:\n"
            "  • Request a copy of your credit report from the relevant bureau\n"
            "  • Have this decision reviewed by a human under GDPR Article 22\n"
            "  • Contact our team at appeals@example.com\n\n"
            "This notice is provided in accordance with the Equal Credit Opportunity "
            "Act (Regulation B)."
        )
        letter_sv = (
            "Tack för din ansökan. Efter noggrann granskning kan vi inte godkänna "
            "din kreditbegäran just nu. Huvudskälen är:\n\n"
            f"{reasons_text}\n\n"
            "Du har rätt att:\n"
            "  • Begära en kopia av din kreditupplysning\n"
            "  • Få beslutet granskat av en människa enligt GDPR artikel 22\n"
            "  • Kontakta vårt team på appeals@example.com"
        )
        final_decision = "DECLINE"
    
    new_audit = append_audit_step(
        state,
        agent="explainer (mock)",
        action=f"generated_{letter_type.lower()}_letter",
        output_summary=f"letter_type={letter_type}",
        tools_called=["translate_feature_to_plain"] if letter_type == "DECLINE" else [],
    )
    
    return {
        "customer_letter_en": letter_en,
        "customer_letter_sv": letter_sv,
        "final_decision": final_decision,
        "audit_trail": new_audit,
    }
