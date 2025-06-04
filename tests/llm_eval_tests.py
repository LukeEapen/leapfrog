import pytest
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric
)

# Example test data (replace with your real input/output from agents)
EXAMPLE_AGENT_CALLS = [
    {
        "agent_id": "agent_4_1",
        "input_text": """Feature Name: Create an Account

Feature Overview:
The Create an Account feature delivers comprehensive capabilities for establishing and managing charged-off accounts within the trade credit ecosystem, leveraging a modern credit card core platform. It supports both frontbook account creation—provisioning new accounts with unique identifiers, configurable products, and terms—and backbook account reactivation by importing historical data to preserve account continuity. This feature also enables linking existing payment instruments securely from legacy systems and assigning multiple participants with defined roles to facilitate collaborative account management. Additionally, it supports flexible trade credit product configurations, including risk models, fees, and a specialized 90-day same-as-cash promotional product with automated term conversion.

This feature encompasses the full onboarding workflow for new customers, integrating seamlessly with acquisition, credit decisioning, and compliance systems to ensure real-time, compliant account setup. It excludes low-level transaction processing and dispute management, which are handled by other platform components. Designed as part of a cloud-native, API-first architecture, it aligns with PCI DSS standards and embeds risk and compliance controls, supporting strategic goals of operational control, agility, and enhanced customer experience in credit product lifecycle management.""",
        "response_text": """User Story 1
As a credit operations specialist, I want to create new charged-off accounts with unique identifiers and configurable product terms so that I can onboard customers efficiently while maintaining accurate account records.

Acceptance Criteria:

System generates a unique account identifier upon creation.
User can select and configure product options including risk models and fees during account setup.
Account creation triggers validation against credit decisioning and compliance systems in real-time..""",
        "context": "This product is a modern, cloud-native credit card core platform designed specifically for the U.S. financial services sector, focusing on credit card management. It serves as a comprehensive solution for credit product lifecycle management, targeting internal stakeholders such as credit operations, risk management, compliance teams, and customer service units. Delivered via an API-first architecture, the platform supports real-time issuance of both virtual and physical credit cards, credit line assignments, transaction authorization, billing, payments, rewards, dispute resolution, and delinquency workflows. By replacing legacy third-party systems, it empowers the organization with full control over credit issuance, account management, and servicing, enabling seamless integration with internal tools and customer-facing applications."
    },
    # Add more test cases for agent_4_2, 4_3, etc.
]

def test_llm_agent_outputs():
    for call in EXAMPLE_AGENT_CALLS:
        test_case = LLMTestCase(
            input=call["input_text"],
            actual_output=call["response_text"],
            context=call["context"],
            expected_output=None  # Optional: can provide if testing strict correctness
        )

        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            HallucinationMetric(threshold=0.7),
            BiasMetric(threshold=0.7)
        ]

        assert_test(test_case, metrics)
