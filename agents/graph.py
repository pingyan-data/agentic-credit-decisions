"""
agents/graph.py
---------------
LangGraph orchestration: assemble the 3 agents into a state machine with
conditional routing.

The graph topology:

    START
      │
      ▼
   underwriter
      │
      ▼
   compliance       (always runs after underwriter)
      │
      ▼
   [router_after_compliance]
      │
      ├── if compliance flagged or protected_class → explainer (forced REVIEW)
      └── else → explainer (whatever band the underwriter chose)
      │
      ▼
     END

Conditional routing happens INSIDE explainer_node based on state, but we
also expose the routing decision in the graph topology for visualization.
"""

from langgraph.graph import StateGraph, START, END

from agents.state import CreditDecisionState
from agents.underwriter import underwriter_node, underwriter_node_mock
from agents.compliance import compliance_node, compliance_node_mock
from agents.explainer import explainer_node, explainer_node_mock


def build_graph(use_mock: bool = False):
    """
    Build the credit decision LangGraph.
    
    Args:
        use_mock: if True, use mock agents (no API calls, deterministic).
                  Use this for unit tests, offline runs, and CI.
    
    Returns:
        Compiled LangGraph app.
    """
    if use_mock:
        uw = underwriter_node_mock
        co = compliance_node_mock
        ex = explainer_node_mock
    else:
        uw = underwriter_node
        co = compliance_node
        ex = explainer_node
    
    graph = StateGraph(CreditDecisionState)
    
    # ─── Add nodes ────────────────────────────────────────────────────
    graph.add_node("underwriter", uw)
    graph.add_node("compliance", co)
    graph.add_node("explainer", ex)
    
    # ─── Add edges ────────────────────────────────────────────────────
    graph.add_edge(START, "underwriter")
    graph.add_edge("underwriter", "compliance")
    graph.add_edge("compliance", "explainer")
    graph.add_edge("explainer", END)
    
    return graph.compile()


def run_application(
    application_id: str,
    application: dict,
    use_mock: bool = False,
) -> CreditDecisionState:
    """
    Convenience function: build the graph, run an application through it,
    return the final state.
    
    Args:
        application_id: unique identifier for this application
        application: dict of borrower features matching model schema
        use_mock: use mock agents (no LLM calls)
    
    Returns:
        Final CreditDecisionState with all agent outputs and full audit trail.
    """
    from agents.state import make_initial_state
    
    app = build_graph(use_mock=use_mock)
    initial_state = make_initial_state(application_id, application)
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick smoke test with mock agents
    import json
    
    test_application = {
        "loan_amnt": 10000,
        "int_rate": 12.5,
        "annual_inc": 60000,
        "dti": 0.18,
        "delinq_2yrs": 0,
        "revol_util": 0.45,
        "open_acc": 8,
        "total_acc": 15,
        # ... add other features matching your model schema
    }
    
    result = run_application(
        application_id="TEST_001",
        application=test_application,
        use_mock=True,
    )
    
    print("\n" + "=" * 60)
    print(f"Application: {result['application_id']}")
    print(f"PD score: {result['pd_score']}")
    print(f"Decision band: {result['decision_band']}")
    print(f"Final decision: {result['final_decision']}")
    print(f"Compliance: {result['compliance_status']}")
    print("\n--- Reasoning ---")
    print(result["underwriter_reasoning"])
    print("\n--- Customer letter (EN) ---")
    print(result["customer_letter_en"])
    print("\n--- Audit trail ---")
    for step in result["audit_trail"]:
        print(f"  [{step['timestamp']}] {step['agent']}: {step['action']}")
    print("=" * 60)
