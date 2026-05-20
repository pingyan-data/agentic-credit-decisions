# EU AI Act — Consumer Credit as High-Risk AI System

## Source
Regulation (EU) 2024/1689 of the European Parliament and of the Council on harmonised rules on artificial intelligence (Artificial Intelligence Act). Published in the Official Journal of the European Union, July 12, 2024. Effective in phases through 2026-2027.

## Why this applies to consumer credit underwriting

### Annex III — High-Risk AI Systems
Annex III, point 5(b) explicitly classifies as high-risk:

> "AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score, with the exception of AI systems used for the purpose of detecting financial fraud."

This means any AI system that produces a probability of default, credit score, or recommended credit decision for a natural person (consumer) is a HIGH-RISK system under the Act.

## Key obligations for high-risk credit AI systems

### Article 9 — Risk management system
A continuous risk management system must be established, implemented, documented, and maintained. This includes identifying foreseeable risks (including bias and unintended outcomes) and adopting risk mitigation measures.

### Article 10 — Data governance
Training, validation, and testing data sets must be relevant, representative, and free of errors. Data sets must take into account characteristics specific to the geographical, behavioural, or functional setting in which the system will be used. Bias detection and mitigation is mandatory.

### Article 13 — Transparency and provision of information to deployers
The high-risk AI system must be sufficiently transparent to enable deployers to interpret the system's output and use it appropriately. Instructions for use must include the system's intended purpose, level of accuracy, and known limitations.

### Article 14 — Human oversight
High-risk AI systems must be designed such that they can be effectively overseen by natural persons during the period the system is in use. Human overseers must be able to:
- Fully understand the capacities and limitations of the system
- Remain aware of the possible tendency to over-rely on system outputs (automation bias)
- Correctly interpret the system's output
- Decide not to use the system or otherwise disregard, override, or reverse the output

### Article 15 — Accuracy, robustness and cybersecurity
The system must achieve an appropriate level of accuracy, robustness, and cybersecurity. Performance metrics and their levels must be declared.

## Implications for credit decisions

For a credit decision system to comply with the EU AI Act:

1. **Explainability is mandatory** — The system must produce interpretable outputs (SHAP values, top features, plain-language reasoning).
2. **Human override must be possible** — The decision band thresholds (approve / review / decline) must allow human underwriters to override automated recommendations, especially in edge cases.
3. **Bias monitoring is required** — Regular audits for disparate impact across protected classes (or their proxies, such as zip code as a proxy for ethnicity) must be performed and documented.
4. **Audit trails must be maintained** — Each decision, the model version used, the inputs, the SHAP values, and any human intervention must be logged and retrievable for audit.

## Penalties for non-compliance

Article 99 of the AI Act establishes administrative fines:
- Up to EUR 35 million or 7% of total worldwide annual turnover for non-compliance with prohibited AI practices
- Up to EUR 15 million or 3% of total worldwide annual turnover for non-compliance with high-risk system obligations
- Up to EUR 7.5 million or 1% of total worldwide annual turnover for supplying incorrect, incomplete, or misleading information

These figures are based on whichever amount is higher.
