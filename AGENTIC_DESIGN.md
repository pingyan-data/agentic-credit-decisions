# Agentic Credit Decisions
## Multi-Agent Orchestration for Consumer Credit Underwriting

> **An extension of `llm-credit-risk` —— from PD model to multi-agent decision system.**

---

## 1. Why this project

我的 `llm-credit-risk` repo 已经有完整的 PD model + SHAP + decision policy。但它是一个**单一模型**:输入特征,输出概率。

真实的信贷决策流程不是这样的。一笔申请要经过:
1. **Underwriting**:模型给出风险评分和初始建议
2. **Compliance review**:监管要求(EU AI Act、Reg B、GDPR)是否满足
3. **Customer communication**:决策需要被解释给客户(尤其是拒绝时,法律要求 Adverse Action Notice)

每一步都需要不同的**专业判断**和**工具**。这是一个天然的 multi-agent 编排问题。

这个项目把上面三步建模成 3 个 specialized AI agent,用 **LangGraph** 编排。

---

## 2. Why this matters for Klarna interviews

我在 Section 5 笔记里已经分析过:**2030 年 Klarna 可能会再裁员至少1000 人，整体规模缩到 <2,000 人,留下来的是 judgment + governance + relationship 角色**。公司应该关心的核心能力是:

| 能力 | 这个项目怎么体现 |
|---|---|
| **Judgment layer**(决定模型何时该被覆盖) | Underwriter agent + 3 个分支决策(approve/review/decline) |
| **Model governance**(监管对接) | Compliance agent + EU AI Act / Reg B 的 RAG 检索 |
| **Cross-functional translation**(技术 → 业务/客户) | Explainer agent 把 SHAP 翻成自然语言 AAN |
| **AI orchestration**(管理 agent fleet) | LangGraph state machine + conditional routing |
| **Domain expertise**(信贷业务深度) | 三个 agent 的 prompt 里都嵌入了真实信贷术语和 regulatory citation |

To summarize my agnetic system design:
> "I built a 3-agent system that simulates Klarna's underwriting workflow: an underwriter agent that calls my existing PD model as a tool, a compliance agent that uses RAG to check EU AI Act and Reg B, and an explainer agent that generates Adverse Action Notices. The orchestration is LangGraph with conditional routing based on PD score and compliance status."

---

## 3. Architecture

### 3.1 Overall flow

```
                    ┌──────────────────────┐
                    │  Borrower Application│
                    │  (LendingClub format) │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Underwriter Agent   │
                    │  ──────────────────  │
                    │  Tools:              │
                    │   • get_pd_score()   │ ← wraps existing 05_pd_model
                    │   • get_shap_values()│ ← wraps existing 06_explainability
                    │   • check_policy()   │ ← wraps decision band logic
                    │  Output: initial decision + reasoning
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Compliance Agent    │
                    │  ──────────────────  │
                    │  Tools:              │
                    │   • search_regulations() │ ← RAG over EU AI Act, Reg B, GDPR
                    │   • check_protected_class()│ ← bias detection
                    │  Output: compliance status + required disclosures
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Explainer Agent     │
                    │  ──────────────────  │
                    │  Tools:              │
                    │   • format_aan()     │ ← Reg B Adverse Action template
                    │   • translate_features()│ ← SHAP → plain language
                    │  Output: customer-facing letter (EN + SV)
                    └──────────┬───────────┘
                               │
                  ┌────────────┼────────────┐
                  ↓            ↓            ↓
              APPROVE      HUMAN REVIEW   DECLINE
            + Welcome     + Queue ticket  + AAN
```

### 3.2 LangGraph state machine

**Why LangGraph(not CrewAI / AutoGen)**:
- LangGraph 是 LangChain 同一家,跟我的 GDPR RAG 项目栈一致
- 显式 state machine,每个 node 和 edge 可以独立测试 → 适合金融 ML 的 reproducibility 要求
- Conditional routing 支持复杂分支(我们需要 PD score-based routing)
- 内置 checkpointing → 可以 replay,符合 audit 要求

**State 定义**:
```python
class CreditDecisionState(TypedDict):
    # Input
    application: dict              # raw borrower application
    
    # Underwriter outputs
    pd_score: float
    decision_band: str             # 'approve' | 'review' | 'decline'
    underwriter_reasoning: str
    shap_top_features: list[dict]  # [{name, value, contribution}]
    
    # Compliance outputs
    compliance_status: str         # 'pass' | 'flagged' | 'fail'
    compliance_citations: list[str]
    required_disclosures: list[str]
    
    # Explainer outputs
    customer_letter_en: str
    customer_letter_sv: str        # Swedish version (Klarna home market)
    
    # Final
    final_decision: str
    audit_trail: list[dict]        # every agent step logged with timestamp
```

### 3.3 Conditional routing

```
START
  │
  ▼
[underwriter_node]
  │
  ▼
[router: PD score]──────┬──────────────┐
                        │              │
                  PD < 0.10        PD ≥ 0.30
                  fast approve   straight decline
                        │              │
                        ▼              ▼
                [compliance_node]   [compliance_node]
                  │                    │
                  ▼                    ▼
                [explainer:approve] [explainer:decline]
                  │                    │
                  └──────────┬─────────┘
                             ▼
                            END

  0.10 ≤ PD < 0.30 (review band)
                        │
                        ▼
                [compliance_node]
                        │
                        ▼
                [router: compliance flag]
                        │
                  ┌─────┴─────┐
                pass        flagged
                  │            │
                  ▼            ▼
            human_review    explainer:decline
                              with bias notice
```

---

## 4. The three agents

### 4.1 Underwriter Agent

**Role**: 风险评估专家。给一笔申请 → 给 PD score + 决策带 + reasoning。

**Tools**:
- `get_pd_score(borrower_features)` —— 调用现有 model_C(challenger)+ calibrator
- `get_shap_values(borrower_features)` —— 调用现有 SHAP explainer
- `check_decision_policy(pd_score)` —— 根据 risk appetite 返回 approve/review/decline

**Prompt 核心**:
> "你是 Klarna 的 senior underwriter。基于 PD model 输出和 SHAP top drivers,给出**初始决策建议**和**3 句话以内的 reasoning**。重点关注:模型置信度(校准过的 PD)、关键风险驱动因素、是否有 SHAP 中显示的非常规模式。**不要**编造模型没有给的数字。"

**关键设计**:
- Agent **不预测 PD**,而是**调用 tool 获得 PD**。这避免 LLM 幻觉,体现 deterministic ML model 是 source of truth
- 输出必须 cite 至少一个 SHAP feature 作为 reasoning 依据
- 如果 PD score 在 0.10–0.30 (review band),agent 必须明确说"recommend manual review by senior underwriter"

### 4.2 Compliance Agent

**Role**: 合规专家。检查决策是否符合监管要求。

**Tools**:
- `search_regulations(query)` —— RAG over compliance docs (EU AI Act, ECOA Reg B, GDPR Art. 22)
- `check_protected_class_features(shap_features)` —— 检查 SHAP top drivers 是否包含受保护属性的代理变量
- `check_explainability_threshold(model_complexity)` —— EU AI Act high-risk system 要求

**Prompt 核心**:
> "你是 Klarna 的 model risk + compliance officer。审查 underwriter 的决策。重点检查:(1) EU AI Act 把消费信贷归为 high-risk system,要求 explainability;(2) Reg B 要求拒绝必须给出 specific reasons,且不得依赖受保护属性;(3) GDPR Art. 22 给客户人工 review 的权利。**Cite specific regulatory clauses.**"

**关键设计**:
- 这是项目的 **RAG component** —— 直接复用我 GDPR RAG repo 的 architecture
- Compliance docs 放在 `data/compliance_docs/`(我会准备 3 份精简版)
- 输出必须有 **regulatory citation**(监管条款引用),不能只说"this is compliant"

### 4.3 Explainer Agent

**Role**: 客户沟通专家。把技术决策翻译成客户能看懂的信件。

**Tools**:
- `format_adverse_action_notice(decision, top_features)` —— Reg B 模板
- `translate_shap_to_plain_language(features)` —— SHAP → 自然语言映射表
- `translate_to_swedish(text)` —— 瑞典语翻译(Klarna 本土)

**Prompt 核心**:
> "你是 Klarna 的客户沟通专家。基于 underwriter 决策和 compliance 检查结果,生成发给客户的信件。**Approve**:简短欢迎信,说明信用额度。**Decline**:Adverse Action Notice,必须 (1) 说明决策、(2) 列出 4 个具体原因(从 SHAP top features 来,翻译成日常语言)、(3) 告知客户如何 contact bureau 和申诉。**Review**:中性的"我们正在审核"信件,设定 expectation。语调:专业、empathetic、避免 financial jargon。"

**关键设计**:
- AAN 必须 strict follow Reg B 格式 —— 这是合规硬约束
- SHAP 翻译表(`feature_glossary.csv`)把模型特征名翻成自然语言:
  - `dti` → "your monthly debt is high relative to your income"
  - `revol_util` → "you are using a high portion of your available credit"
  - `delinq_2yrs` → "you have recent missed payments on other accounts"
- 双语输出体现 Klarna 的 EU 多语言现实

---

## 5. File structure

```
agentic-credit-decisions/
├── README.md                              # The portfolio-facing pitch
├── AGENTIC_DESIGN.md                      # This file
├── requirements_agents.txt                # langgraph, langchain, etc.
│
├── agents/
│   ├── __init__.py
│   ├── state.py                           # CreditDecisionState definition
│   ├── tools.py                           # Wrappers around existing PD model
│   ├── underwriter.py                     # Underwriter agent
│   ├── compliance.py                      # Compliance agent
│   ├── explainer.py                       # Explainer agent
│   └── graph.py                           # LangGraph orchestration
│
├── data/
│   └── compliance_docs/                   # RAG corpus
│       ├── eu_ai_act_high_risk.md         # EU AI Act, Annex III, point 5(b)
│       ├── reg_b_adverse_action.md        # ECOA Reg B § 1002.9
│       ├── gdpr_art_22.md                 # GDPR Art. 22 automated decisions
│       └── feature_glossary.csv           # SHAP → plain language mapping
│
├── scenarios/                             # Test cases
│   ├── 01_clear_approve.json              # PD ~0.05, clean profile
│   ├── 02_borderline_review.json          # PD ~0.20, ambiguous
│   ├── 03_clear_decline.json              # PD ~0.45, multiple risk factors
│   └── 04_protected_class_proxy.json      # Triggers compliance flag
│
├── app_agentic.py                         # Streamlit demo
└── tests/
    ├── test_underwriter.py
    ├── test_compliance.py
    └── test_full_workflow.py
```

---


## 6. What this project does NOT claim to do

目前的局限性和真实性:

✗ **不**是生产级系统(没有 Klarna 真实数据、没有真实 LLM provider redundancy、没有真实 EU AI Act 法律意见)
✗ **不**自动 retrain 模型(static model artifacts,不是 ML training pipeline)
✗ **不**是真实合规审查(citation 是 demo,真实合规需要 lawyer review)
✗ **不**取代任何现有 underwriting 系统

✓ **是**一个**完整的端到端 agent orchestration demo**
✓ **是**对真实 underwriting workflow 的**结构化建模**
✓ **是**我**理解** judgment + governance + translation 三层价值的**证据**

---

## 8. Summary 

> "After reading Klarna's engineering blog and the CEO's 20VC interview, I noticed a pattern: the data scientists who survive the AI compression are the ones doing judgment, governance, and cross-functional translation — not the ones writing models faster. So I built a project that demonstrates I think in those three layers.
>
> The base is my LLM credit risk repo, which has a calibrated PD model with SHAP. On top of that, I added a 3-agent orchestration layer using LangGraph. The first agent is an underwriter that calls the PD model as a tool — note that the model is the source of truth, the agent doesn't predict, it interprets. The second is a compliance agent doing RAG over EU AI Act and Reg B Adverse Action requirements. This bridges directly to my GDPR RAG project. The third is an explainer that translates SHAP outputs into customer-facing letters.
>
> The thing I'm most proud of is the conditional routing in LangGraph: PD < 0.10 goes one path, the 0.10–0.30 review band goes through human escalation, PD ≥ 0.30 with protected-class proxy detection takes a third path. Every transition is logged for audit.
>
> The project doesn't claim to be production-ready. It's a structured statement that I understand where data scientists add value when AI absorbs the routine work."

---

## 9. Stretch goals(time permitting)

If I have time after the 2-week core build:

1. **Add evaluation framework**: 50 synthetic scenarios with ground-truth labels, measure agent agreement vs expected decisions. Adds rigor.
2. **Add async logging to a SQLite "audit DB"**: every agent step persisted with full state. Useful for the "reproducibility" argument.
3. **Add a "challenger agent"**: a 4th agent that argues the opposite of the underwriter, forcing explicit disagreement resolution. This is the 2025 Anthropic-style "constitutional AI" pattern applied to underwriting.
4. **Translate one scenario fully into Swedish** end-to-end and screenshot it. Klarna interviews will love this.

---

**Bottom line**: 这个项目是 orchestration、governance、translation 三层的具体演示。
