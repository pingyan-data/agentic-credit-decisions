"""
app_agentic.py
--------------
Streamlit demo for the agentic credit decision system.

Run with:
    streamlit run app_agentic.py

This adds a new tab to the existing dashboard, but works standalone too.
"""

import json
from pathlib import Path

import streamlit as st

from agents.graph import run_application


st.set_page_config(
    page_title="Agentic Credit Decisions",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Agentic Credit Decision System")
st.caption(
    "Multi-agent orchestration: Underwriter → Compliance → Explainer. "
    "Built on top of the llm-credit-risk PD model."
)

# ─── Sidebar: scenario picker and execution mode ──────────────────────────
with st.sidebar:
    st.header("Configuration")
    
    use_mock = st.toggle(
        "Offline mode (mock agents)",
        value=True,
        help="Mock mode runs the full pipeline without calling any LLM API. "
             "Useful for demos and tests.",
    )
    
    scenarios_dir = Path(__file__).parent / "scenarios"
    scenario_files = sorted(scenarios_dir.glob("*.json"))
    scenario_names = [f.stem for f in scenario_files]
    
    selected_scenario = st.selectbox(
        "Test scenario",
        options=scenario_names,
        index=0,
    )
    
    selected_path = scenarios_dir / f"{selected_scenario}.json"
    scenario = json.loads(selected_path.read_text())
    
    st.divider()
    st.markdown("**Scenario description:**")
    st.write(scenario.get("description", ""))
    
    if "expected_band" in scenario:
        st.markdown("**Expected outcomes:**")
        st.write(f"- Band: `{scenario.get('expected_band')}`")
        st.write(f"- Compliance: `{scenario.get('expected_compliance')}`")
        st.write(f"- Final: `{scenario.get('expected_final')}`")
    
    st.divider()
    run_button = st.button("▶ Run pipeline", type="primary", use_container_width=True)


# ─── Main panel: results ──────────────────────────────────────────────────
if not run_button:
    st.info("Select a scenario in the sidebar and click **Run pipeline** to start.")
    st.markdown("### Application data preview")
    st.json(scenario["application"])
    st.stop()


with st.spinner("Running 3-agent pipeline..."):
    result = run_application(
        application_id=scenario["application_id"],
        application=scenario["application"],
        use_mock=use_mock,
    )

# ─── Top-level decision badge ────────────────────────────────────────────
decision = result["final_decision"]
band = result["decision_band"]
compliance = result["compliance_status"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("PD score (calibrated)", f"{result['pd_score']:.2%}")

with col2:
    color = {"approve": "🟢", "review": "🟡", "decline": "🔴"}[band]
    st.metric("Underwriter band", f"{color} {band}")

with col3:
    color = {"pass": "🟢", "flagged": "🟡", "fail": "🔴"}[compliance]
    st.metric("Compliance", f"{color} {compliance}")

with col4:
    final_color = {"APPROVE": "🟢", "HUMAN_REVIEW": "🟡", "DECLINE": "🔴"}[decision]
    st.metric("Final decision", f"{final_color} {decision}")

st.divider()


# ─── Three columns: each agent's contribution ─────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Underwriter", "⚖️ Compliance", "✉️ Customer Letter", "📋 Audit Trail"]
)

with tab1:
    st.subheader("Underwriter Agent output")
    
    st.markdown("**Reasoning:**")
    st.info(result["underwriter_reasoning"])
    
    st.markdown("**Top SHAP features:**")
    
    for feat in result["shap_top_features"]:
        emoji = "🔺" if feat["direction"] == "increases_pd" else "🔻"
        st.markdown(
            f"{emoji} **{feat['name']}** = {feat['value']} "
            f"(SHAP contribution: {feat['contribution']:+.4f})"
        )

with tab2:
    st.subheader("Compliance Agent output")
    
    st.markdown("**Status:** " + compliance)
    
    if result.get("protected_class_concern"):
        st.error("⚠️ Protected class proxy detected in top features")
    
    st.markdown("**Regulatory citations:**")
    for citation in result["compliance_citations"]:
        st.markdown(f"- {citation}")
    
    st.markdown("**Required disclosures:**")
    for disclosure in result["required_disclosures"]:
        st.markdown(f"- {disclosure}")

with tab3:
    st.subheader("Explainer Agent output")
    
    col_en, col_sv = st.columns(2)
    
    with col_en:
        st.markdown("**🇬🇧 English version**")
        st.text_area(
            label="letter_en",
            value=result["customer_letter_en"],
            height=400,
            label_visibility="collapsed",
        )
    
    with col_sv:
        st.markdown("**🇸🇪 Swedish version**")
        st.text_area(
            label="letter_sv",
            value=result["customer_letter_sv"],
            height=400,
            label_visibility="collapsed",
        )

with tab4:
    st.subheader("Audit trail")
    st.caption(
        "Every agent step is logged with a timestamp, the action taken, and the "
        "tools called. This supports the EU AI Act audit requirement."
    )
    
    for i, step in enumerate(result["audit_trail"], 1):
        with st.expander(f"Step {i}: {step['agent']} — {step['action']}"):
            st.text(f"Timestamp: {step['timestamp']}")
            st.text(f"Output: {step['output_summary']}")
            st.text(f"Tools called: {', '.join(step['tools_called']) or '(none)'}")
