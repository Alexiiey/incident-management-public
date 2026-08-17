"""Incident analysis engine for the VIP Transport Quality Control system.

Provides a deterministic, keyword/regex-based classification engine that
works instantly with no external dependency, plus an optional LLM-backed
engine that is used automatically when an API key is supplied. The rule
based engine is the primary path so the Streamlit demo never depends on a
paid API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Domain knowledge base
# --------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Punctuality & No-Show": [
        "late", "delay", "delayed", "no-show", "no show", "did not show",
        "didn't show", "waited", "waiting", "not on time",
    ],
    "Vehicle Cleanliness & Condition": [
        "dirty", "smell", "odor", "odour", "stain", "unclean", "messy",
        "damaged", "broken", "old car", "condition", "scratch",
    ],
    "Driver Behavior & Professionalism": [
        "rude", "unprofessional", "impolite", "disrespectful", "aggressive",
        "shouted", "yelled", "argument", "attitude", "inappropriate",
    ],
    "Safety & Security Incident": [
        "accident", "crash", "collision", "injury", "injured", "police",
        "arrested", "assault", "harassment", "unsafe", "speeding",
        "reckless", "hospital", "emergency", "fire", "theft", "stolen",
    ],
    "Route & Navigation": [
        "wrong route", "got lost", "lost", "detour", "gps", "navigation",
        "wrong way", "longer route", "wrong address",
    ],
    "Communication & Booking": [
        "no response", "did not answer", "didn't answer", "unreachable",
        "booking error", "wrong pickup", "miscommunication",
        "confirmation", "double booked", "cancelled without notice",
    ],
    "Amenities & In-Car Service": [
        "water", "wifi", "wi-fi", "charger", "amenities", "temperature",
        "air conditioning", " ac ", "music", "newspaper",
    ],
    "Billing & Pricing Dispute": [
        "overcharged", "billing", "invoice", "price", "refund", "payment",
        "charged twice", "extra fee", "hidden fee",
    ],
}

DEFAULT_CATEGORY = "General Service Issue"

CRITICAL_KEYWORDS = [
    "accident", "crash", "collision", "injury", "injured", "police",
    "arrested", "assault", "harassment", "hospital", "emergency", "fire",
    "theft", "stolen", "life-threatening", "critical condition",
]
HIGH_KEYWORDS = [
    "no-show", "no show", "did not show", "didn't show",
    "cancelled last minute", "major delay", "very late", "extremely late",
    "unsafe", "reckless", "speeding", "rude", "shouted", "yelled",
    "aggressive", "damaged vehicle", "lost luggage", "left behind",
]
MEDIUM_KEYWORDS = [
    "late", "delay", "dirty", "unprofessional", "wrong route", "lost",
    "no response", "overcharged", "cold", "uncomfortable", "smell",
    "messy", "impolite",
]
LOW_KEYWORDS = [
    "minor", "slightly", "small issue", "little late", "not a big deal",
    "suggestion", "could be better", "small delay",
]

SEVERITY_TIERS: list[tuple[str, list[str]]] = [
    ("Critical", CRITICAL_KEYWORDS),
    ("High", HIGH_KEYWORDS),
    ("Medium", MEDIUM_KEYWORDS),
    ("Low", LOW_KEYWORDS),
]

VIP_KEYWORDS = [
    "vip", "celebrity", "ceo", "chairman", "diplomat", "ambassador",
    "royal", "high-profile", "high profile", "board member", "executive",
    "repeat client", "platinum", "premium client", "key account",
]

BASE_FINANCIAL_IMPACT_EUR: dict[str, float] = {
    "Critical": 2500.0,
    "High": 800.0,
    "Medium": 250.0,
    "Low": 60.0,
}

CATEGORY_IMPACT_MULTIPLIER: dict[str, float] = {
    "Safety & Security Incident": 1.8,
    "Punctuality & No-Show": 1.2,
    "Vehicle Cleanliness & Condition": 1.0,
    "Driver Behavior & Professionalism": 1.3,
    "Route & Navigation": 1.0,
    "Communication & Booking": 1.1,
    "Amenities & In-Car Service": 0.6,
    "Billing & Pricing Dispute": 1.4,
    DEFAULT_CATEGORY: 1.0,
}

VIP_IMPACT_MULTIPLIER = 1.5

DRIVER_ACTIONS_BY_CATEGORY: dict[str, list[str]] = {
    "Punctuality & No-Show": [
        "Review the driver's live GPS log against the scheduled pickup time.",
        "Conduct a coaching session on buffer-time planning and traffic anticipation.",
    ],
    "Vehicle Cleanliness & Condition": [
        "Send the vehicle for immediate detailing before its next assignment.",
        "Retrain the driver on the pre-shift cleanliness checklist.",
    ],
    "Driver Behavior & Professionalism": [
        "Open a formal conduct review with the driver and HR.",
        "Enroll the driver in the client-etiquette refresher course.",
    ],
    "Safety & Security Incident": [
        "Suspend the driver from active duty pending investigation.",
        "Collect a full written statement and dashcam footage.",
    ],
    "Route & Navigation": [
        "Verify the driver's navigation app is updated with real-time traffic data.",
        "Review preferred VIP routes with the driver for the client's key destinations.",
    ],
    "Communication & Booking": [
        "Audit the driver's response time on the dispatch app.",
        "Confirm the driver received the booking confirmation and pickup instructions.",
    ],
    "Amenities & In-Car Service": [
        "Check the vehicle's amenities kit against the standard inventory list.",
        "Remind the driver of the pre-trip amenities checklist.",
    ],
    "Billing & Pricing Dispute": [
        "Cross-check the trip log against the invoice generated for the client.",
        "Confirm no manual overrides were applied to the fare without authorization.",
    ],
    DEFAULT_CATEGORY: [
        "Review the trip log and driver notes for additional context.",
        "Schedule a debrief with the driver on the reported issue.",
    ],
}

FLEET_ACTIONS_BY_CATEGORY: dict[str, list[str]] = {
    "Punctuality & No-Show": [
        "Adjust dispatch buffer times for this route/time slot.",
        "Flag the booking window for review by Operations.",
    ],
    "Vehicle Cleanliness & Condition": [
        "Escalate to the fleet maintenance team for inspection.",
        "Add the vehicle to the rotation for a full valet before its next VIP trip.",
    ],
    "Driver Behavior & Professionalism": [
        "Log the incident in the driver's performance file.",
        "Review client-facing conduct standards at the next fleet briefing.",
    ],
    "Safety & Security Incident": [
        "Notify Legal and Insurance departments immediately.",
        "Report to the relevant regulatory authority if required by law.",
    ],
    "Route & Navigation": [
        "Update the fleet's preferred-route database with the correct itinerary.",
        "Check for a wider navigation/GPS issue across the fleet.",
    ],
    "Communication & Booking": [
        "Audit the dispatch system for the root cause of the miscommunication.",
        "Review booking confirmation workflow with the reservations team.",
    ],
    "Amenities & In-Car Service": [
        "Replenish amenities stock across the affected vehicle category.",
        "Update the amenities checklist if a systemic gap is identified.",
    ],
    "Billing & Pricing Dispute": [
        "Escalate to Finance for invoice reconciliation.",
        "Review pricing rules for the affected route/service tier.",
    ],
    DEFAULT_CATEGORY: [
        "Log the incident for trend analysis in the next quality review.",
    ],
}

CLIENT_RESPONSE_BY_VIP_RISK: dict[str, str] = {
    "Critical": (
        "Immediate personal call from the Operations Director within 1 hour, "
        "a formal written apology, a full refund of the affected trip, and a "
        "complimentary upgrade on the client's next 3 bookings."
    ),
    "High": (
        "Call from the Client Relations Manager within 4 hours, a written "
        "apology, and a service credit equivalent to 50% of the trip cost."
    ),
    "Medium": (
        "Email apology within 24 hours from Client Relations, plus a service "
        "credit voucher valid on the next booking."
    ),
    "Low": (
        "Standard acknowledgement email, logged for quality tracking. No "
        "compensation required unless the client explicitly requests it."
    ),
}


# --------------------------------------------------------------------------
# Result model
# --------------------------------------------------------------------------

@dataclass
class IncidentAnalysis:
    """Structured result of an incident analysis."""

    severity: str
    category: str
    financial_impact_eur: float
    vip_risk: str
    confidence: str
    matched_keywords: list[str] = field(default_factory=list)
    driver_actions: list[str] = field(default_factory=list)
    fleet_actions: list[str] = field(default_factory=list)
    client_response: str = ""
    engine_used: str = "Rule-Based Engine"


# --------------------------------------------------------------------------
# Rule-based engine
# --------------------------------------------------------------------------

def _find_matches(text: str, keywords: list[str]) -> list[str]:
    """Return the subset of `keywords` found in `text` (case-insensitive)."""
    lowered = f" {text.lower()} "
    return [kw for kw in keywords if kw.lower() in lowered]


def _classify_category(text: str) -> tuple[str, list[str]]:
    """Pick the category with the most keyword matches."""
    best_category = DEFAULT_CATEGORY
    best_matches: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = _find_matches(text, keywords)
        if len(matches) > len(best_matches):
            best_category, best_matches = category, matches
    return best_category, best_matches


def _classify_severity(text: str) -> tuple[str, list[str], str]:
    """Pick the highest-priority severity tier with at least one match."""
    for severity, keywords in SEVERITY_TIERS:
        matches = _find_matches(text, keywords)
        if matches:
            return severity, matches, "High"
    return "Medium", [], "Low"


def _classify_vip_risk(text: str, severity: str) -> tuple[str, list[str]]:
    vip_matches = _find_matches(text, VIP_KEYWORDS)
    is_vip_mentioned = bool(vip_matches)

    if is_vip_mentioned and severity in ("Critical", "High"):
        risk = "Critical"
    elif is_vip_mentioned:
        risk = "High"
    elif severity == "Critical":
        risk = "High"
    elif severity == "High":
        risk = "Medium"
    else:
        risk = "Low"
    return risk, vip_matches


def _estimate_financial_impact(severity: str, category: str, vip_risk: str) -> float:
    base = BASE_FINANCIAL_IMPACT_EUR[severity]
    multiplier = CATEGORY_IMPACT_MULTIPLIER.get(category, 1.0)
    impact = base * multiplier
    if vip_risk in ("Critical", "High"):
        impact *= VIP_IMPACT_MULTIPLIER
    return round(impact, -1)  # round to nearest 10 EUR


def build_action_plan(category: str, severity: str, vip_risk: str) -> tuple[list[str], list[str], str]:
    """Assemble driver/fleet actions and the client response for a case."""
    driver_actions = list(DRIVER_ACTIONS_BY_CATEGORY.get(category, DRIVER_ACTIONS_BY_CATEGORY[DEFAULT_CATEGORY]))
    fleet_actions = list(FLEET_ACTIONS_BY_CATEGORY.get(category, FLEET_ACTIONS_BY_CATEGORY[DEFAULT_CATEGORY]))

    if severity == "Critical":
        fleet_actions.append("Open a formal incident file and notify senior management within 24 hours.")
    elif severity == "High":
        fleet_actions.append("Flag the case for review at the next weekly quality meeting.")

    client_response = CLIENT_RESPONSE_BY_VIP_RISK[vip_risk]
    return driver_actions, fleet_actions, client_response


def analyze_incident_rule_based(text: str) -> IncidentAnalysis:
    """Classify an incident report using the keyword/regex rules engine."""
    text = re.sub(r"\s+", " ", text or "").strip()

    category, category_matches = _classify_category(text)
    severity, severity_matches, confidence = _classify_severity(text)
    vip_risk, vip_matches = _classify_vip_risk(text, severity)
    financial_impact = _estimate_financial_impact(severity, category, vip_risk)
    driver_actions, fleet_actions, client_response = build_action_plan(category, severity, vip_risk)

    if not text:
        confidence = "Low"

    matched_keywords = sorted(set(category_matches + severity_matches + vip_matches))

    return IncidentAnalysis(
        severity=severity,
        category=category,
        financial_impact_eur=financial_impact,
        vip_risk=vip_risk,
        confidence=confidence,
        matched_keywords=matched_keywords,
        driver_actions=driver_actions,
        fleet_actions=fleet_actions,
        client_response=client_response,
        engine_used="Rule-Based Engine",
    )


# --------------------------------------------------------------------------
# Optional LLM-backed engine
# --------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a quality control analyst for a luxury chauffeur \
(NCC / VIP transport) company. Analyze the incident report and respond with \
STRICT JSON only, matching this schema:
{
  "severity": "Low" | "Medium" | "High" | "Critical",
  "category": string,
  "financial_impact_eur": number,
  "vip_risk": "Low" | "Medium" | "High" | "Critical",
  "driver_actions": [string, ...],
  "fleet_actions": [string, ...],
  "client_response": string
}
Do not include any text outside the JSON object."""


def analyze_incident_llm(text: str, api_key: str, model: str = "gpt-4o-mini") -> IncidentAnalysis:
    """Classify an incident using an OpenAI model. Raises on any failure so
    the caller can fall back to the rule-based engine."""
    from openai import OpenAI  # local import: optional dependency

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    payload = json.loads(response.choices[0].message.content)

    return IncidentAnalysis(
        severity=payload["severity"],
        category=payload["category"],
        financial_impact_eur=float(payload["financial_impact_eur"]),
        vip_risk=payload["vip_risk"],
        confidence="High",
        matched_keywords=[],
        driver_actions=list(payload.get("driver_actions", [])),
        fleet_actions=list(payload.get("fleet_actions", [])),
        client_response=payload.get("client_response", ""),
        engine_used=f"LLM ({model})",
    )


def analyze_incident(text: str, use_llm: bool = False, api_key: str | None = None) -> IncidentAnalysis:
    """Analyze an incident report, using the LLM engine if requested and
    available, otherwise the deterministic rule-based engine."""
    if use_llm and api_key:
        try:
            return analyze_incident_llm(text, api_key)
        except Exception:
            result = analyze_incident_rule_based(text)
            result.engine_used = "Rule-Based Engine (LLM call failed, fallback used)"
            return result
    return analyze_incident_rule_based(text)
