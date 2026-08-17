"""Generate a synthetic dataset of VIP transport quality/incident records.

Running this script writes `data/incidents.csv`, a deterministic (seeded)
dataset of 50 incidents used by the Streamlit dashboard demo so the app
works immediately after cloning, with no manual data entry required.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from incident_engine import CATEGORY_KEYWORDS, DEFAULT_CATEGORY

SEED = 42
N_INCIDENTS = 50
OUTPUT_PATH = Path(__file__).parent / "data" / "incidents.csv"

DRIVERS = [
    "Marco Bellini", "Julien Fontaine", "Ahmed El-Sayed", "Thomas Reeves",
    "Luca Moretti", "Nikolai Petrov", "James Whitfield", "Diego Alvarez",
    "Klaus Richter", "Antoine Dubois", "Rafael Costa", "Viktor Novak",
]

VEHICLES = [
    "Mercedes S-Class", "BMW 7 Series", "Bentley Flying Spur",
    "Range Rover Autobiography", "Audi A8", "Rolls-Royce Ghost",
    "Mercedes V-Class (Van)",
]

CLIENT_TIERS = ["VIP", "Corporate", "Standard"]
CLIENT_TIER_WEIGHTS = [0.35, 0.35, 0.30]

CATEGORIES = list(CATEGORY_KEYWORDS.keys()) + [DEFAULT_CATEGORY]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
SEVERITY_WEIGHTS = [0.35, 0.35, 0.22, 0.08]

AUDITORS = ["Sophie Lambert", "Marc Renard", "Elena Rossi", "Daniel Kim"]

DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "Punctuality & No-Show": [
        "Driver arrived 20 minutes late to the airport pickup.",
        "Client waited outside the hotel for over 30 minutes, driver was delayed in traffic.",
        "No-show reported for the 07:00 pickup, client had to book alternative transport.",
    ],
    "Vehicle Cleanliness & Condition": [
        "Client reported a strong smell inside the vehicle and stains on the seats.",
        "Interior was dirty with visible dust on the dashboard.",
        "Exterior of the vehicle was not washed, arrived visibly dirty for VIP pickup.",
    ],
    "Driver Behavior & Professionalism": [
        "Client complained the driver was rude and dismissive during the ride.",
        "Driver was on the phone for most of the trip, appeared unprofessional.",
        "Client reported an argument with the driver over the route taken.",
    ],
    "Safety & Security Incident": [
        "Minor collision reported while merging onto the highway, no injuries.",
        "Driver was reported for reckless speeding on a wet road.",
        "Police were called after a dispute at the pickup location.",
    ],
    "Route & Navigation": [
        "Driver got lost on the way to the venue, added 25 minutes to the trip.",
        "GPS directed the driver the wrong way, client missed part of the meeting.",
        "Driver took a much longer route than necessary, client questioned the detour.",
    ],
    "Communication & Booking": [
        "Driver did not respond to the client's calls before pickup.",
        "Booking confirmation was never sent, client was unsure of pickup time.",
        "Wrong pickup address was used due to a miscommunication with dispatch.",
    ],
    "Amenities & In-Car Service": [
        "No water bottles available in the vehicle as requested.",
        "Wifi in the vehicle was not working during the client's business call.",
        "Air conditioning was not adjusted to the client's preference.",
    ],
    "Billing & Pricing Dispute": [
        "Client was overcharged compared to the quoted fare.",
        "Invoice included an unexplained extra fee.",
        "Client requested a refund after being charged twice for the same trip.",
    ],
    DEFAULT_CATEGORY: [
        "General feedback submitted after the trip, no specific issue detailed.",
        "Client left a neutral review mentioning a minor inconvenience.",
        "Miscellaneous quality note logged by the auditor during a spot check.",
    ],
}

VIP_PREFIXES = [
    "VIP client feedback: ", "Repeat platinum client reported: ",
    "Executive client (CEO) noted: ", "",
]


def _weighted_choice(options: list[str], weights: list[float]) -> str:
    return random.choices(options, weights=weights, k=1)[0]


def generate_incidents(n: int = N_INCIDENTS, seed: int = SEED) -> pd.DataFrame:
    """Build a deterministic synthetic incident dataset."""
    rng = random.Random(seed)
    start_date = datetime.now() - timedelta(days=180)

    rows = []
    for i in range(1, n + 1):
        category = rng.choice(CATEGORIES)
        severity = rng.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]
        description = rng.choice(DESCRIPTION_TEMPLATES[category])
        vip_prefix = rng.choice(VIP_PREFIXES)
        driver = rng.choice(DRIVERS)
        vehicle = rng.choice(VEHICLES)
        client_tier = rng.choices(CLIENT_TIERS, weights=CLIENT_TIER_WEIGHTS, k=1)[0]
        auditor = rng.choice(AUDITORS)
        incident_date = start_date + timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23))

        base_impact = {"Low": 60, "Medium": 250, "High": 800, "Critical": 2500}[severity]
        financial_impact = round(base_impact * rng.uniform(0.7, 1.4), -1)

        resolved = rng.random() > 0.15
        resolution_hours = round(rng.uniform(1, 72), 1) if resolved else None

        rows.append(
            {
                "incident_id": f"INC-{i:04d}",
                "date": incident_date.strftime("%Y-%m-%d %H:%M"),
                "driver_name": driver,
                "vehicle_type": vehicle,
                "client_tier": client_tier,
                "category": category,
                "severity": severity,
                "description": f"{vip_prefix}{description}",
                "financial_impact_eur": financial_impact,
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
