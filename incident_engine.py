"""Incident analysis engine for the VIP Transport Quality Control system.

Provides a deterministic, keyword/regex-based classification engine that
works instantly with no external dependency, plus an optional LLM-backed
engine that is used automatically when an API key is supplied. The rule
based engine is the primary path so the Streamlit demo never depends on a
paid API key.

Every client served by this operation is a premium/VIP-tier client by
business definition, so risk is expressed as "Client Risk" (driven by
incident severity and reputational-escalation signals) rather than by
detecting whether a given client happens to be VIP.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------
# Domain knowledge base
# --------------------------------------------------------------------------

VEHICLE_TYPES: list[str] = [
    "Mercedes S-Class",
    "BMW 7 Series",
    "Audi A8",
    "Range Rover Autobiography",
    "Bentley Flying Spur",
    "Rolls-Royce Ghost",
    "Mercedes V-Class (Van)",
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Driver Behavior": [
        "rude", "unprofessional", "impolite", "disrespectful", "aggressive",
        "shouted", "yelled", "argument", "attitude", "inappropriate",
        "on the phone", "distracted",
    ],
    "Booking Error": [
        "wrong pickup", "wrong address", "booking error", "double booked",
        "cancelled last minute", "cancelled without notice", "no confirmation",
        "miscommunication", "wrong date", "wrong time", "no response",
        "did not answer", "didn't answer", "unreachable",
    ],
    "Procedure Missing": [
        "procedure", "protocol", "checklist", "sop", "not followed",
        "missing paperwork", "no name board", "no meet and greet",
        "pre-trip check", "confirmation call", "documentation was missing",
    ],
    "Punctuality": [
        "late", "delay", "delayed", "no-show", "no show", "did not show",
        "didn't show", "waited", "waiting", "not on time", "extremely late",
        "behind schedule",
    ],
    "Customer Experience": [
        "disappointed", "expectations", "overall experience", "neglected",
        "complaint", "subpar", "not satisfied", "could be better",
        "negative experience", "felt rushed",
    ],
    "Safety": [
        "accident", "crash", "collision", "injury", "injured", "police",
        "arrested", "assault", "harassment", "unsafe", "speeding",
        "reckless", "hospital", "emergency", "fire", "theft", "stolen",
    ],
    "Vehicle Quality": [
        "dirty", "smell", "odor", "odour", "stain", "unclean", "messy",
        "damaged", "broken", "old car", "condition", "scratch",
        "mechanical", "breakdown", "broke down", "flat tire",
    ],
}

DEFAULT_CATEGORY = "Customer Experience"

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
    "broke down", "breakdown",
]
MEDIUM_KEYWORDS = [
    "late", "delay", "dirty", "unprofessional", "wrong route", "lost",
    "no response", "overcharged", "cold", "uncomfortable", "smell",
    "messy", "impolite", "miscommunication", "not followed",
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

# Signals that escalate reputational risk further — every client is already
# premium/VIP, so these represent *additional* exposure (media, legal,
# high-profile status) rather than a baseline VIP/non-VIP split.
ESCALATION_KEYWORDS = [
    "media", "press", "lawsuit", "legal action", "public relations",
    "social media", "posted online", "viral", "board member", "diplomat",
    "ambassador", "celebrity", "high-profile", "high profile",
    "repeat client", "key account", "escalated", "regulator",
]

CLIENT_RISK_ORDER = ["Low", "Medium", "High", "Critical"]

BASE_FINANCIAL_IMPACT_EUR: dict[str, float] = {
    "Critical": 2500.0,
    "High": 800.0,
    "Medium": 250.0,
    "Low": 60.0,
}

COMPENSATION_RATE_BY_SEVERITY: dict[str, float] = {
    "Critical": 1.5,
    "High": 0.75,
    "Medium": 0.35,
    "Low": 0.15,
}

CATEGORY_IMPACT_MULTIPLIER: dict[str, float] = {
    "Driver Behavior": 1.3,
    "Booking Error": 1.1,
    "Procedure Missing": 1.0,
    "Punctuality": 1.2,
    "Customer Experience": 1.0,
    "Safety": 1.8,
    "Vehicle Quality": 1.0,
}

CLIENT_RISK_IMPACT_MULTIPLIER = 1.25
DEFAULT_TRIP_VALUE_EUR = 200.0

DRIVER_ACTIONS_BY_CATEGORY: dict[str, list[str]] = {
    "Driver Behavior": [
        "Open a formal conduct review with the driver and HR.",
        "Enroll the driver in the client-etiquette refresher course.",
    ],
    "Booking Error": [
        "Confirm the driver received the correct booking details from dispatch.",
        "Debrief the driver on the booking discrepancy.",
    ],
    "Procedure Missing": [
        "Retrain the driver on the standard operating procedure that was skipped.",
        "Confirm the driver has the current SOP checklist for this service type.",
    ],
    "Punctuality": [
        "Review the driver's live GPS log against the scheduled pickup time.",
        "Conduct a coaching session on buffer-time planning and traffic anticipation.",
    ],
    "Customer Experience": [
        "Review the trip log and driver notes for additional context.",
        "Schedule a debrief with the driver on the client's feedback.",
    ],
    "Safety": [
        "Suspend the driver from active duty pending investigation.",
        "Collect a full written statement and dashcam footage.",
    ],
    "Vehicle Quality": [
        "Send the vehicle for immediate detailing or inspection before its next assignment.",
        "Retrain the driver on the pre-shift vehicle checklist.",
    ],
}

FLEET_ACTIONS_BY_CATEGORY: dict[str, list[str]] = {
    "Driver Behavior": [
        "Log the incident in the driver's performance file.",
        "Review client-facing conduct standards at the next fleet briefing.",
    ],
    "Booking Error": [
        "Audit the dispatch system for the root cause of the booking error.",
        "Review the booking confirmation workflow with the reservations team.",
    ],
    "Procedure Missing": [
        "Audit compliance with the standard procedure across the fleet for this service type.",
        "Update the SOP checklist if a systemic gap is identified.",
    ],
    "Punctuality": [
        "Adjust dispatch buffer times for this route/time slot.",
        "Flag the booking window for review by Operations.",
    ],
    "Customer Experience": [
        "Log the incident for trend analysis in the next quality review.",
        "Review overall service standards for this client tier.",
    ],
    "Safety": [
        "Notify Legal and Insurance departments immediately.",
        "Report to the relevant regulatory authority if required by law.",
    ],
    "Vehicle Quality": [
        "Escalate to the fleet maintenance team for inspection.",
        "Add the vehicle to the rotation for service before its next VIP trip.",
    ],
}

CLIENT_RESPONSE_BY_CLIENT_RISK: dict[str, str] = {
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
    client_risk: str
    confidence: str
    trip_value_eur: float = 0.0
    financial_impact_eur: float = 0.0
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


def _classify_client_risk(text: str, severity: str) -> tuple[str, list[str]]:
    """Client risk starts from severity (every client is premium-tier) and
    escalates by one level if reputational-escalation signals are present."""
    escalation_matches = _find_matches(text, ESCALATION_KEYWORDS)
    risk = severity if severity in CLIENT_RISK_ORDER else "Medium"
    if escalation_matches and risk != "Critical":
        idx = CLIENT_RISK_ORDER.index(risk)
        risk = CLIENT_RISK_ORDER[min(idx + 1, len(CLIENT_RISK_ORDER) - 1)]
    return risk, escalation_matches


def compute_trip_value(
    cost_rules_df: pd.DataFrame | None,
    vehicle: str | None,
    service_type: str,
    disposal_hours: float | None,
    extra_services: list[str] | None,
) -> float:
    """Estimate the value of the underlying trip from the cost rate card."""
    if cost_rules_df is None or cost_rules_df.empty or not vehicle:
        return DEFAULT_TRIP_VALUE_EUR

    rule_type = "Disposal" if service_type == "At Disposal (hourly)" else "Transfer"
    match = cost_rules_df[
        (cost_rules_df["service_type"] == rule_type) & (cost_rules_df["item"] == vehicle)
    ]
    if match.empty:
        base = DEFAULT_TRIP_VALUE_EUR
    elif rule_type == "Disposal":
        base = float(match["cost_eur"].iloc[0]) * max(disposal_hours or 1, 1)
    else:
        base = float(match["cost_eur"].iloc[0])

    extra_cost = 0.0
    if extra_services:
        extra_rows = cost_rules_df[
            (cost_rules_df["service_type"] == "Other Service")
            & (cost_rules_df["item"].isin(extra_services))
        ]
        extra_cost = float(extra_rows["cost_eur"].sum())

    return round(base + extra_cost, 2)


def _estimate_financial_impact(severity: str, category: str, client_risk: str, trip_value: float) -> float:
    """Compensation estimate = trip value x severity rate x category weight,
    escalated further for High/Critical client risk."""
    rate = COMPENSATION_RATE_BY_SEVERITY[severity]
    multiplier = CATEGORY_IMPACT_MULTIPLIER.get(category, 1.0)
    impact = trip_value * rate * multiplier
    if client_risk in ("Critical", "High"):
        impact *= CLIENT_RISK_IMPACT_MULTIPLIER
    return round(impact, -1)  # round to nearest 10 EUR


def build_action_plan(category: str, severity: str, client_risk: str) -> tuple[list[str], list[str], str]:
    """Assemble driver/fleet actions and the client response for a case."""
    driver_actions = list(DRIVER_ACTIONS_BY_CATEGORY.get(category, DRIVER_ACTIONS_BY_CATEGORY[DEFAULT_CATEGORY]))
    fleet_actions = list(FLEET_ACTIONS_BY_CATEGORY.get(category, FLEET_ACTIONS_BY_CATEGORY[DEFAULT_CATEGORY]))

    if severity == "Critical":
        fleet_actions.append("Open a formal incident file and notify senior management within 24 hours.")
    elif severity == "High":
        fleet_actions.append("Flag the case for review at the next weekly quality meeting.")

    client_response = CLIENT_RESPONSE_BY_CLIENT_RISK[client_risk]
    return driver_actions, fleet_actions, client_response


def analyze_incident_rule_based(
    text: str,
    vehicle: str | None = None,
    service_type: str = "Transfer",
    disposal_hours: float | None = None,
    extra_services: list[str] | None = None,
    cost_rules_df: pd.DataFrame | None = None,
) -> IncidentAnalysis:
    """Classify an incident report using the keyword/regex rules engine."""
    text = re.sub(r"\s+", " ", text or "").strip()

    category, category_matches = _classify_category(text)
    severity, severity_matches, confidence = _classify_severity(text)
    client_risk, escalation_matches = _classify_client_risk(text, severity)
    trip_value = compute_trip_value(cost_rules_df, vehicle, service_type, disposal_hours, extra_services)
    financial_impact = _estimate_financial_impact(severity, category, client_risk, trip_value)
    driver_actions, fleet_actions, client_response = build_action_plan(category, severity, client_risk)

    if not text:
        confidence = "Low"

    matched_keywords = sorted(set(category_matches + severity_matches + escalation_matches))

    return IncidentAnalysis(
        severity=severity,
        category=category,
        client_risk=client_risk,
        confidence=confidence,
        trip_value_eur=trip_value,
        financial_impact_eur=financial_impact,
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
(NCC / VIP transport) company where every client is premium-tier. Analyze the \
incident report and respond with STRICT JSON only, matching this schema:
{
  "severity": "Low" | "Medium" | "High" | "Critical",
  "category": string,
  "client_risk": "Low" | "Medium" | "High" | "Critical",
  "driver_actions": [string, ...],
  "fleet_actions": [string, ...],
  "client_response": string
}
"client_risk" reflects reputational/business risk given the severity and any
escalation signals (media, legal, high-profile status) — not whether the
client is VIP, since all clients already are. Do not include any text
outside the JSON object."""


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
        client_risk=payload["client_risk"],
        confidence="High",
        matched_keywords=[],
        driver_actions=list(payload.get("driver_actions", [])),
        fleet_actions=list(payload.get("fleet_actions", [])),
        client_response=payload.get("client_response", ""),
        engine_used=f"LLM ({model})",
    )


def analyze_incident(
    text: str,
    vehicle: str | None = None,
    service_type: str = "Transfer",
    disposal_hours: float | None = None,
    extra_services: list[str] | None = None,
    cost_rules_df: pd.DataFrame | None = None,
    use_llm: bool = False,
    api_key: str | None = None,
) -> IncidentAnalysis:
    """Analyze an incident report, using the LLM engine if requested and
    available, otherwise the deterministic rule-based engine. The trip value
    (and therefore the financial impact) always uses the local cost rules."""
    trip_value = compute_trip_value(cost_rules_df, vehicle, service_type, disposal_hours, extra_services)

    if use_llm and api_key:
        try:
            result = analyze_incident_llm(text, api_key)
            result.trip_value_eur = trip_value
            result.financial_impact_eur = _estimate_financial_impact(
                result.severity, result.category, result.client_risk, trip_value
            )
            return result
        except Exception:
            result = analyze_incident_rule_based(
                text, vehicle, service_type, disposal_hours, extra_services, cost_rules_df
            )
            result.engine_used = "Rule-Based Engine (LLM call failed, fallback used)"
            return result

    return analyze_incident_rule_based(
        text, vehicle, service_type, disposal_hours, extra_services, cost_rules_df
    )
