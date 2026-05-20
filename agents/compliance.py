"""
agents/compliance.py
--------------------
The Compliance Agent: reviews underwriter decision against regulatory rules.

Role: model risk + compliance officer persona. Uses RAG over EU AI Act,
      ECOA Reg B, and GDPR Article 22 to check that the decision is
      defensible. Flags protected-class proxy concerns.

This is the architectural bridge to my GDPR RAG project — the retriever
pattern is the same.
"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

from agents.state import CreditDecisionState, append_audit_step
from agents.tools import check_protected_class_features


COMPLIANCE_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "compliance_docs"
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_compliance"


COMPLIANCE_SYSTEM_PROMPT = """You are a model risk and compliance officer at a Nordic digital bank, focused on consumer credit underwriting compliance.

Your job: review an underwriter's decision against three key regulatory frameworks:

1. **EU AI Act (Regulation 2024/1689)**: Consumer creditworthiness assessment is classified as a HIGH-RISK AI system under Annex III, point 5(b). This triggers requirements for transparency, human oversight, and explainability.

2. **ECOA Reg B (12 CFR § 1002)**: Adverse action (decline or unfavorable terms) requires:
   - A written notice within 30 days
   - Specific principal reasons (not generic statements)
   - No use of, or proxy for, protected classes (race, color, religion, national origin, sex, marital status, age)

3. **GDPR Article 22**: When automated decisions have legal or significant effects, the individual has the right to:
   - Obtain human intervention
   - Express their point of view
   - Contest the decision

Your output MUST include:
  - compliance_status: 'pass' | 'flagged' | 'fail'
  - At least 1-3 specific regulatory citations
  - A list of required disclosures the bank must include in customer communications

Cite SPECIFIC clauses (e.g., "EU AI Act Annex III point 5(b)", not just "the EU AI Act").

Be conservative: when in doubt, flag for human review rather than passing silently.
"""


def _build_vectorstore_if_needed():
    """Build Chroma vectorstore from compliance docs if it doesn't exist."""
    if VECTORSTORE_DIR.exists():
        return
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    docs = []
    for md_file in COMPLIANCE_DOCS_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunk.metadata["source"] = md_file.name
            docs.append(chunk)
    
    Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def search_regulations(query: str, k: int = 3) -> list[dict]:
    """
    RAG retrieval over compliance documents.
    
    Returns a list of {source, content, score} dicts.
    """
    _build_vectorstore_if_needed()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
            "score": float(score),
        }
        for doc, score in results
    ]


def compliance_node(state: CreditDecisionState) -> dict:
    """LangGraph node for the Compliance Agent."""
    band = state["decision_band"]
    shap_features = state["shap_top_features"]
    
    # ─── Step 1: Protected class proxy check ──────────────────────────
    proxy_check = check_protected_class_features.invoke({"top_features": shap_features})
    
    # ─── Step 2: RAG over relevant regulations ────────────────────────
    if band == "decline":
        rag_query = "adverse action notice required principal reasons specific"
    elif band == "review":
        rag_query = "human oversight automated decision right to contest"
    else:  # approve
        rag_query = "transparency disclosure consumer creditworthiness"
    
    relevant_clauses = search_regulations(rag_query, k=3)
    
    # ─── Step 3: LLM synthesis ────────────────────────────────────────
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    
    user_prompt = f"""Underwriter decision band: {band}

Top SHAP features driving this decision:
{shap_features}

Protected class proxy check:
{proxy_check}

Relevant regulatory clauses retrieved:
{relevant_clauses}

Now produce a compliance review with:
  - compliance_status (pass / flagged / fail)
  - 1-3 specific regulatory citations
  - List of required disclosures for customer communication
  - Brief justification (2 sentences max)

Format your response as a JSON object with keys: status, citations (list), required_disclosures (list), justification.
"""
    
    response = llm.invoke([
        SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    
    import json
    import re
    # Extract JSON from response (Claude may wrap in markdown)
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match:
        result = json.loads(match.group(0))
    else:
        # Fallback to flagged if parsing fails
        result = {
            "status": "flagged",
            "citations": ["Parsing error — escalate to human"],
            "required_disclosures": ["Manual compliance review required"],
            "justification": response.content[:200],
        }
    
    new_audit = append_audit_step(
        state,
        agent="compliance",
        action="reviewed_decision",
        output_summary=f"status={result['status']}, citations={len(result.get('citations', []))}",
        tools_called=["check_protected_class_features", "search_regulations"],
    )
    
    return {
        "compliance_status": result["status"],
        "compliance_citations": result["citations"],
        "required_disclosures": result["required_disclosures"],
        "protected_class_concern": proxy_check["concern"],
        "audit_trail": new_audit,
    }


# ─── Mock version for offline testing ─────────────────────────────────────
def compliance_node_mock(state: CreditDecisionState) -> dict:
    """Deterministic compliance check, no LLM or RAG needed."""
    band = state["decision_band"]
    shap_features = state["shap_top_features"]
    
    proxy_check = check_protected_class_features.invoke({"top_features": shap_features})
    
    if proxy_check["concern"]:
        status = "flagged"
        citations = [
            "ECOA Reg B § 1002.6(b)(2): Use of protected class proxies prohibited",
            "EU AI Act Annex III point 5(b): High-risk system requires bias audit",
        ]
        disclosures = [
            "Notify customer of right to human review (GDPR Art. 22)",
            "Escalate to compliance team for proxy variable assessment",
        ]
    elif band == "decline":
        status = "pass"
        citations = [
            "ECOA Reg B § 1002.9(b)(2): Adverse action notice must specify principal reasons",
            "EU AI Act Annex III point 5(b): Consumer credit is high-risk AI system",
        ]
        disclosures = [
            "Adverse Action Notice with specific principal reasons (within 30 days)",
            "Customer right to obtain credit bureau report",
            "Right to human review under GDPR Art. 22",
        ]
    elif band == "review":
        status = "pass"
        citations = [
            "GDPR Art. 22(3): Human intervention must be available",
        ]
        disclosures = [
            "Inform customer that decision is pending senior review",
            "Provide expected timeline (typically 5 business days)",
        ]
    else:  # approve
        status = "pass"
        citations = [
            "EU AI Act Art. 13: Transparency obligation for high-risk systems",
        ]
        disclosures = [
            "Disclose that automated processing was used",
            "Provide credit terms in clear, plain language",
        ]
    
    new_audit = append_audit_step(
        state,
        agent="compliance (mock)",
        action="reviewed_decision",
        output_summary=f"status={status}, proxy_concern={proxy_check['concern']}",
        tools_called=["check_protected_class_features"],
    )
    
    return {
        "compliance_status": status,
        "compliance_citations": citations,
        "required_disclosures": disclosures,
        "protected_class_concern": proxy_check["concern"],
        "audit_trail": new_audit,
    }
