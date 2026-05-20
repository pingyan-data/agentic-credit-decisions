# Agentic Credit Decisions

> A multi-agent orchestration layer on top of `llm-credit-risk`. 3 agents
> (underwriter, compliance, explainer) coordinated by LangGraph, simulating
> a Nordic digital bank's consumer credit underwriting workflow.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)]()
[![LangChain](https://img.shields.io/badge/RAG-LangChain%20+%20Chroma-green.svg)]()

---

## What this is

A working demonstration that I think about credit ML the way 2030-era data
scientists need to: not as a model-training exercise, but as a
**judgment + governance + translation** orchestration problem.

```
   Borrower Application
          │
          ▼
   ┌─────────────────┐
   │  Underwriter    │  ← Calls existing PD model + SHAP as tools
   │  Agent          │  ← Output: decision band + reasoning
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │  Compliance     │  ← RAG over EU AI Act + Reg B + GDPR Art 22
   │  Agent          │  ← Output: regulatory citations + required disclosures
   └─────────────────┘
          │
          ▼
   ┌─────────────────┐
   │  Explainer      │  ← SHAP → plain-language Adverse Action Notice
   │  Agent          │  ← Output: customer letter (EN + SV)
   └─────────────────┘
          │
          ▼
   APPROVE | HUMAN_REVIEW | DECLINE
```

## Why this design

**1. The model is a tool, not a black box.**
The Underwriter Agent does not predict PD. It calls `get_pd_score()` — a
deterministic wrapper around the calibrated XGBoost model from
`llm-credit-risk`. This avoids LLM hallucination on numeric values and
makes outputs reproducible for audit.

**2. Compliance is RAG, not a hardcoded rulebook.**
Regulations change. The Compliance Agent retrieves relevant clauses from
3 source documents (EU AI Act high-risk system requirements, ECOA
Regulation B Adverse Action rules, GDPR Article 22 automated
decision-making rights), letting the orchestration adapt as the corpus
grows.

**3. Customer communication is the last mile.**
The Explainer Agent generates Adverse Action Notices that strictly follow
ECOA Reg B § 1002.9 format, in both English and Swedish (Klarna's home
market). Customer-facing communication is exactly the cross-functional
translation work that AI-augmentation makes more valuable, not less.

**4. Every step is logged.**
The audit trail captures every agent's action, tools called, and output
summary with timestamps. This supports the EU AI Act Article 12
record-keeping obligation for high-risk systems.

## Quick start

```bash
# Clone alongside llm-credit-risk
git clone https://github.com/pingyan-data/agentic-credit-decisions.git
cd agentic-credit-decisions

# Install agent dependencies (assumes llm-credit-risk env exists)
pip install -r requirements_agents.txt

# Run a test scenario in offline mode (no API key needed)
python -m agents.graph

# Or launch the Streamlit demo
streamlit run app_agentic.py
```

## File structure

```
agentic-credit-decisions/
├── README.md                    # this file
├── AGENTIC_DESIGN.md            # detailed design rationale
├── requirements_agents.txt
│
├── agents/
│   ├── state.py                 # LangGraph state TypedDict
│   ├── tools.py                 # Wrappers for PD model + SHAP
│   ├── underwriter.py           # Underwriter agent (real + mock)
│   ├── compliance.py            # Compliance agent + RAG retriever
│   ├── explainer.py             # Explainer agent + AAN generator
│   └── graph.py                 # LangGraph orchestration
│
├── data/
│   └── compliance_docs/
│       ├── eu_ai_act_high_risk.md
│       ├── reg_b_adverse_action.md
│       ├── gdpr_art_22.md
│       └── feature_glossary.csv
│
├── scenarios/                   # 4 test cases
│   ├── 01_clear_approve.json
│   ├── 02_borderline_review.json
│   ├── 03_clear_decline.json
│   └── 04_protected_class_proxy.json
│
└── app_agentic.py               # Streamlit demo
```

## Test scenarios

| # | Scenario | PD | Expected Final Decision |
|---|---|---|---|
| 01 | Clean profile, low DTI, no delinquencies | ~0.05 | APPROVE |
| 02 | Borderline: elevated DTI + utilization | ~0.20 | HUMAN_REVIEW |
| 03 | High-risk: low income, multiple delinquencies | ~0.45 | DECLINE (full AAN) |
| 04 | Decline-band PD with zip_code as top SHAP feature | ~0.35 | HUMAN_REVIEW (proxy flag) |

## What this project does NOT claim to be

- Not a production-grade system (no real-time API, no bank-grade
  reliability, no red team review)
- Not legal compliance advice (regulatory citations are illustrative; real
  compliance review requires a lawyer)
- Not a replacement for the existing `llm-credit-risk` PD model — it
  **uses** that model, it doesn't replace it

## What this project IS

- A 600-line, end-to-end demonstration of multi-agent orchestration
  applied to a real financial decision workflow
- Evidence that I understand where data science adds durable value:
  judgment, governance, and cross-functional translation
- A bridge between my GDPR RAG project (compliance retrieval) and my
  llm-credit-risk project (PD model)

## Tech stack

- **Orchestration**: LangGraph 0.2 (state machine + conditional routing)
- **LLM**: Anthropic Claude Haiku 4.5 (fast, cheap, deterministic at temp=0)
- **RAG**: LangChain + ChromaDB + sentence-transformers
- **Existing model**: XGBoost + SHAP + isotonic calibration (from
  `llm-credit-risk`)
- **Demo**: Streamlit

## License

MIT — Ping Yan, 2026.

## Related work

- [`llm-credit-risk`](https://github.com/pingyan-data/llm-credit-risk) —
  the underlying PD model with LLM feature extraction
- [`gdpr-chatbot-pipeline`](https://github.com/pingyan-data/gdpr-chatbot-pipeline) —
  the RAG architecture this builds on
- [`sweden-data-job-market`](https://github.com/pingyan-data/sweden-data-job-market) —
  operationally maintained ETL pipeline (related portfolio piece)
