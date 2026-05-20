# GDPR Article 22 — Automated Individual Decision-Making

## Source
Regulation (EU) 2016/679 (General Data Protection Regulation), Article 22.

## The right (Article 22(1))

The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her.

## Application to credit decisions

A credit decision (approve, decline, set credit limit, set interest rate) made solely by an algorithm without meaningful human review falls squarely within Article 22, because it produces:
- Legal effects: a credit contract or its denial
- Similarly significant effects: financial standing, ability to make major purchases, credit history reporting

## Exceptions (Article 22(2))

Article 22(1) does NOT apply if the decision:
- (a) is necessary for entering into, or performance of, a contract between the data subject and a data controller
- (b) is authorised by Union or Member State law to which the controller is subject and which also lays down suitable measures to safeguard the data subject's rights and freedoms and legitimate interests
- (c) is based on the data subject's explicit consent

Most credit decisions fall under exception (a) — necessary for entering into a credit contract. However, the controller MUST still implement safeguards.

## Required safeguards (Article 22(3))

In cases referred to in points (a) and (c) of paragraph 2, the data controller shall implement suitable measures to safeguard the data subject's rights and freedoms and legitimate interests, AT LEAST the right to:
1. Obtain human intervention on the part of the controller
2. Express his or her point of view
3. Contest the decision

## Sensitive data (Article 22(4))

Decisions referred to in paragraph 2 shall not be based on special categories of personal data referred to in Article 9(1) (race, ethnic origin, political opinions, religious beliefs, etc.) unless point (a) or (g) of Article 9(2) applies and suitable measures to safeguard the data subject's rights and freedoms and legitimate interests are in place.

## Practical implications for credit decisions

For a credit underwriting system to comply with Article 22:

1. **Disclose automated processing**: At application time, the customer must be informed that automated decision-making is involved, including meaningful information about the logic involved (Article 13(2)(f), Article 14(2)(g)).

2. **Provide route to human review**: The decision communication (whether approval, decline, or pending review) must inform the customer of their right to request human review. This must be operationally feasible — a real human underwriter must actually review the case if requested.

3. **Allow expression of point of view**: The customer must be able to provide additional context or correct erroneous data that influenced the decision.

4. **Allow contestation**: There must be a process for the customer to formally appeal the decision.

5. **Maintain audit trail**: The decision logic, inputs, model outputs, and any human review must be documented to demonstrate compliance.

## Interaction with EU AI Act

The EU AI Act (Regulation 2024/1689) reinforces and extends GDPR Article 22 for credit AI systems. Where GDPR establishes individual rights, the AI Act establishes systemic obligations on the deployer (the bank). Both apply concurrently.

Specifically, the AI Act's Article 14 (human oversight) requires that humans can:
- Fully understand the system
- Override the decision
- Decide not to use the system in a particular case

This goes beyond GDPR's "human intervention upon request" and requires baseline human oversight capability built into the operational design.

## Recommended communication language

When informing customers of an automated decision:

> "Your application was reviewed using automated processing. You have the right under EU GDPR Article 22 to request human review of this decision, to express your point of view, and to contest the outcome. To exercise these rights, please contact [appeals contact]."

This single sentence covers all three Article 22(3) safeguards.
