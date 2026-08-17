"""Generate a synthetic dataset of VIP transport quality/incident records.

Running this script writes `data/incidents.csv`, a deterministic (seeded)
dataset of 50 incidents used by the Streamlit dashboard demo so the app
works immediately after cloning, with no manual data entry required.

Each row's severity, category, trip value, financial impact and client risk
are computed by running the generated description text through the SAME
rule-based engine used by the live "Incident Analyzer" tab
(`incident_engine.analyze_incident_rule_based`), so the demo dataset and the
live analyzer are always consistent with one another.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from incident_engine import CATEGORY_KEYWORDS, VEHICLE_TYPES, analyze_incident_rule_based

SEED = 42
N_INCIDENTS = 50
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "incidents.csv"
COST_RULES_PATH = DATA_DIR / "cost_rules.csv"

DRIVERS = [
    "Marco Bellini", "Julien Fontaine", "Ahmed El-Sayed", "Thomas Reeves",
    "Luca Moretti", "Nikolai Petrov", "James Whitfield", "Diego Alvarez",
    "Klaus Richter", "Antoine Dubois", "Rafael Costa", "Viktor Novak",
]

CLIENT_TIERS = ["VIP", "Corporate", "Standard"]
CLIENT_TIER_WEIGHTS = [0.55, 0.30, 0.15]

CATEGORIES = list(CATEGORY_KEYWORDS.keys())
SEVERITIES = ["Low", "Medium", "High", "Critical"]
SEVERITY_WEIGHTS = [0.35, 0.35, 0.22, 0.08]

AUDITORS = ["Sophie Lambert", "Marc Renard", "Elena Rossi", "Daniel Kim"]

ESCALATION_PREFIXES = [
    "Repeat client noted: ", "Executive client (CEO) noted: ",
    "Key account feedback: ", "",
]

# description templates, organized by category then intended severity.
# "Medium" is populated for every category and used as the fallback when a
# given (category, severity) pair has no dedicated template.
DESCRIPTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "Driver Behavior": {
        "Low": ["Client felt the driver's attitude was slightly off during the ride."],
        "Medium": [
            "Client complained the driver was rude and dismissive during the ride.",
            "Driver was on the phone for most of the trip, appeared unprofessional.",
        ],
        "High": [
            "Client reported the driver shouted during an argument over the route taken.",
            "Driver was aggressive and disrespectful when the client raised a concern.",
        ],
        "Critical": [
            "Driver became aggressive and the client had to call for police assistance after a heated argument.",
        ],
    },
    "Booking Error": {
        "Low": ["A wrong pickup time was noted initially, a minor booking mismatch resolved quickly."],
        "Medium": [
            "Wrong pickup address was used due to a miscommunication with dispatch.",
            "No confirmation was sent for the booking, client was unsure of the pickup time.",
        ],
        "High": [
            "Trip was cancelled last minute without notice, client had to arrange alternative transport.",
            "Double booked slot meant no vehicle was available for the client's confirmed time.",
        ],
        "Critical": [
            "A booking error caused the client to miss a flight, resulting in a missed connection and an emergency rebooking.",
        ],
    },
    "Procedure Missing": {
        "Low": ["Driver did not display the name board as per the standard meet and greet procedure."],
        "Medium": [
            "Pre-trip vehicle checklist was not completed before the pickup.",
            "Driver skipped the standard confirmation call before arrival.",
        ],
        "High": [
            "Standard procedure was not followed for a high-profile pickup, protocol was not respected.",
            "Required documentation was missing at pickup, procedure not followed.",
        ],
        "Critical": [
            "Safety procedure was not followed and led to an emergency situation during the transfer.",
        ],
    },
    "Punctuality": {
        "Low": ["Small delay of a few minutes, client mentioned it as a minor issue."],
        "Medium": [
            "Driver arrived 20 minutes late to the airport pickup.",
            "Client waited outside the hotel for over 30 minutes, driver was delayed in traffic.",
        ],
        "High": [
            "No-show reported for the 07:00 pickup, client had to book alternative transport.",
            "Driver was extremely late, over an hour behind schedule.",
        ],
        "Critical": [
            "Driver no-show caused the client to miss a critical business flight, resulting in significant financial loss.",
        ],
    },
    "Customer Experience": {
        "Low": [
            "Client left a neutral review mentioning a minor inconvenience.",
            "Client suggested small improvements for future trips.",
        ],
        "Medium": [
            "Client reported feeling generally disappointed with the overall service level.",
            "Overall experience did not meet the client's expectations for this service tier.",
        ],
        "High": [
            "Client reported a very negative overall experience and requested to speak with management.",
            "Client felt neglected throughout the trip and raised a formal complaint.",
        ],
        "Critical": [
            "Client had an extremely negative experience and threatened to escalate the complaint publicly via media.",
        ],
    },
    "Safety": {
        "Low": ["Minor concern raised about the driver following too closely, which felt slightly unsafe."],
        "Medium": [
            "Minor collision reported while merging onto the highway, no injuries.",
            "Client raised concern about the driver speeding on a wet road.",
        ],
        "High": [
            "Driver was reported for reckless speeding on a wet road.",
            "Client reported feeling unsafe due to the driver's erratic maneuvers.",
        ],
        "Critical": [
            "A collision occurred causing an injury, police and emergency services were called to the scene.",
        ],
    },
    "Vehicle Quality": {
        "Low": ["Client noted the vehicle interior was slightly less polished than expected, with a minor scratch visible."],
        "Medium": [
            "Client reported a strong smell inside the vehicle and stains on the seats.",
            "Interior was dirty with visible dust on the dashboard.",
        ],
        "High": [
            "Vehicle broke down during the trip, client had to be transferred to another car.",
            "Exterior of the vehicle was visibly damaged and not fit for a VIP pickup.",
        ],
        "Critical": [
            "A mechanical failure caused the vehicle to break down on the highway during an emergency hospital transfer.",
        ],
    },
}


def _pick_description(rng: random.Random, category: str, severity: str) -> str:
    by_severity = DESCRIPTION_TEMPLATES[category]
    templates = by_severity.get(severity, by_severity["Medium"])
    return rng.choice(templates)


def generate_incidents(n: int = N_INCIDENTS, seed: int = SEED) -> pd.DataFrame:
    """Build a deterministic synthetic incident dataset."""
    rng = random.Random(seed)
    cost_rules_df = pd.read_csv(COST_RULES_PATH)
    extra_options = sorted(cost_rules_df[cost_rules_df["service_type"] == "Other Service"]["item"].unique())
    start_date = datetime.now() - timedelta(days=180)

    rows = []
    for i in range(1, n + 1):
        intended_category = rng.choice(CATEGORIES)
        intended_severity = rng.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]
        description = _pick_description(rng, intended_category, intended_severity)
        prefix = rng.choice(ESCALATION_PREFIXES)
        full_description = f"{prefix}{description}"

        driver = rng.choice(DRIVERS)
        vehicle = rng.choice(VEHICLE_TYPES)
        service_type = rng.choices(["Transfer", "At Disposal (hourly)"], weights=[0.7, 0.3], k=1)[0]
        disposal_hours = rng.choice([2, 3, 4, 5, 6]) if service_type == "At Disposal (hourly)" else None
        n_extras = rng.choice([0, 0, 1, 2])
        extra_services = rng.sample(extra_options, k=n_extras) if extra_options and n_extras else []

        client_tier = rng.choices(CLIENT_TIERS, weights=CLIENT_TIER_WEIGHTS, k=1)[0]
        auditor = rng.choice(AUDITORS)
        incident_date = start_date + timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23))

        analysis = analyze_incident_rule_based(
            full_description,
            vehicle=vehicle,
            service_type=service_type,
            disposal_hours=disposal_hours,
            extra_services=extra_services,
            cost_rules_df=cost_rules_df,
        )

        resolved = rng.random() > 0.15
        resolution_hours = round(rng.uniform(1, 72), 1) if resolved else None

        rows.append(
            {
                "incident_id": f"INC-{i:04d}",
                "date": incident_date.strftime("%Y-%m-%d %H:%M"),
                "driver_name": driver,
                "vehicle_type": vehicle,
                "service_type": service_type,
                "disposal_hours": disposal_hours,
                "client_tier": client_tier,
                "category": analysis.category,
                "severity": analysis.severity,
                "description": full_description,
                "trip_value_eur": analysis.trip_value_eur,
                "financial_impact_eur": analysis.financial_impact_eur,
                "client_risk": analysis.client_risk,
                "resolved": resolved,
                "resolution_time_hours": resolution_hours,
                "auditor": auditor,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = generate_incidents()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic incidents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
