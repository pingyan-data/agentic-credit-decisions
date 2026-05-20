# ECOA Regulation B — Adverse Action Requirements

## Source
Equal Credit Opportunity Act (ECOA), implemented through Regulation B at 12 CFR Part 1002, administered by the Consumer Financial Protection Bureau (CFPB).

## Definition of adverse action

Under § 1002.2(c), "adverse action" means:
- A refusal to grant credit in substantially the amount or on substantially the terms requested in an application
- A termination of an account or an unfavorable change in the terms of an account
- A refusal to increase the amount of credit available to an applicant who has made an application for an increase

## Notice requirement

### § 1002.9(a) — Notification of action taken
A creditor must notify the applicant of action taken on a complete application within 30 days. For adverse action, the notification must contain:
- A statement of the action taken
- The name and address of the creditor
- A statement of the provisions of section 701(a) of the ECOA
- The name and address of the federal agency that administers compliance
- Either:
  - A statement of specific reasons for the action taken; OR
  - A disclosure of the applicant's right to request a statement of reasons

### § 1002.9(b)(2) — Specific reasons
The reasons disclosed must be the principal reasons for the adverse action. Statements that the action was based on the creditor's internal standards or policies, or that the applicant failed to achieve a qualifying score on the creditor's credit scoring system, are insufficient.

## Prohibited bases

Under § 1002.6(b)(2), a creditor shall not consider any of the following in evaluating an application:
- Race, color, religion, national origin, or sex
- Marital status (except as required for community property analysis)
- Age (except as needed to qualify for legal age requirements)
- Receipt of public assistance income
- The fact that the applicant has exercised any right under the Consumer Credit Protection Act

## Proxy variables

A variable that strongly correlates with a prohibited basis can be considered a "proxy" and may itself constitute disparate impact. Common proxies include:
- **Zip code / postal code**: highly correlated with race and ethnicity in segregated areas
- **Surname**: correlated with national origin
- **Educational institution**: correlated with race, national origin, and socioeconomic status
- **Marital status indicators** (titles like Mrs., Mr., Ms., Miss): often capture marital status indirectly

When SHAP analysis reveals that any of these proxies are top drivers of an adverse decision, the creditor must:
1. Document the business necessity of the variable
2. Conduct a less-discriminatory-alternative analysis
3. Consider alternative variables that achieve the same predictive purpose with less disparate impact

## Recommended language for Adverse Action Notices

### What to include (required principal reasons)
Each principal reason should be:
- Specific (not generic)
- Verifiable by the applicant
- Stated in plain language

Examples of acceptable phrasings:
- "Your debt-to-income ratio is higher than our typical threshold for this loan amount"
- "You have one or more recent late payments on existing accounts"
- "Your credit utilization on revolving accounts is high"
- "Your length of credit history is shorter than typical for this product"

### What NOT to say
- "You did not score high enough on our credit scoring system"
- "Our internal policies do not permit approval at this time"
- "You did not meet our credit standards"

These are explicitly insufficient under § 1002.9(b)(2).

## Right to obtain credit report

The applicant has the right to obtain a copy of the credit report used in the decision from the credit reporting agency. The creditor must provide the name, address, and phone number of any consumer reporting agency that provided a report relied upon.

## Right to human review

While Reg B does not explicitly require human review of automated decisions, GDPR Article 22 (applicable to EU residents) does establish this right. For Klarna's EU customer base, the AAN should reference both the right to obtain the credit report and the right to request human review of automated decisions.
