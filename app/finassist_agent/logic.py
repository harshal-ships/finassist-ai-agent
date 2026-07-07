"""Deterministic eligibility rules for the FinAssist demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class LoanStatus(str, Enum):
    PRE_APPROVED = "Pre-Approved"
    NEEDS_REVIEW = "Needs Review"
    STANDARD_PROCESSING = "Standard Processing"


class ClaimStatus(str, Enum):
    HIGH_PRIORITY = "High Priority"
    PREMIUM_MEMBER = "Premium Member"
    STANDARD_CLAIM = "Standard Claim"


@dataclass(frozen=True)
class LoanDecision:
    status: LoanStatus
    action: str
    summary: str


@dataclass(frozen=True)
class ClaimDecision:
    status: ClaimStatus
    action: str
    summary: str


def parse_dollar_amount(text: str) -> float | None:
    """Extract a dollar amount from spoken or written text."""
    if not text:
        return None
    normalized = text.lower().replace(",", "")
    match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand)?", normalized)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)", normalized)
    if not match:
        return None
    value = float(match.group(1))
    if "k" in normalized or "thousand" in normalized:
        value *= 1000
    return value


def evaluate_loan(*, amount: float, annual_income: float, purpose: str = "") -> LoanDecision:
    """Apply FLOW 1 eligibility rules."""
    if annual_income > 50_000 and amount < 20_000:
        return LoanDecision(
            status=LoanStatus.PRE_APPROVED,
            action="Send digital contract",
            summary=(
                f"Loan for ${amount:,.0f} ({purpose or 'general purpose'}) is pre-approved. "
                "We'll email a digital contract to sign."
            ),
        )
    if annual_income < 30_000 or amount > 50_000:
        return LoanDecision(
            status=LoanStatus.NEEDS_REVIEW,
            action="Schedule agent call",
            summary=(
                f"Loan for ${amount:,.0f} needs a specialist review. "
                "We'll schedule a call with a loan agent within one business day."
            ),
        )
    return LoanDecision(
        status=LoanStatus.STANDARD_PROCESSING,
        action="Upload documents",
        summary=(
            f"Loan for ${amount:,.0f} is in standard processing. "
            "Please upload income verification and ID through our secure portal."
        ),
    )


def evaluate_claim(
    *,
    policy_number: str,
    description: str,
    incident_date: str = "",
) -> ClaimDecision:
    """Apply FLOW 2 validation rules."""
    desc = description.lower()
    policy = policy_number.upper().strip()

    if any(word in desc for word in ("injury", "injured", "hospital", "accident")):
        return ClaimDecision(
            status=ClaimStatus.HIGH_PRIORITY,
            action="Assign case manager immediately",
            summary=(
                f"Claim on policy {policy} (incident {incident_date or 'reported today'}) "
                "is high priority. A case manager will contact you within the hour."
            ),
        )
    if policy.startswith("POL-99"):
        return ClaimDecision(
            status=ClaimStatus.PREMIUM_MEMBER,
            action="Fast-track approval",
            summary=(
                f"Premium member claim on {policy} qualifies for fast-track approval. "
                "You'll receive confirmation within 24 hours."
            ),
        )
    return ClaimDecision(
        status=ClaimStatus.STANDARD_CLAIM,
        action="Request photo evidence via WhatsApp",
        summary=(
            f"Standard claim filed on {policy}. "
            "Please send photo evidence via WhatsApp to continue processing."
        ),
    )
