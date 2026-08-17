# VIP Transport — AI Incident & Quality Control System

A Streamlit application that simulates the quality-control and incident-management
workflow of a luxury chauffeur (NCC / VIP transport) operation: live incident
triage powered by an AI classification engine, a fleet-wide operations
analytics dashboard, and an auditable compliance log.

Built to demonstrate the intersection of **operations management**, **quality
auditing**, and **applied AI automation** in a real-world luxury mobility
context.

## Why this project

Running a VIP chauffeur operation means every incident — a late pickup, a
rude interaction, a safety issue — carries reputational and financial risk
with high-value clients. This project shows how a lightweight AI-assisted
triage system can:

- Classify incoming incident reports by **severity** and **category** in
  real time, with no manual tagging.
- Estimate **financial exposure** and **VIP reputational risk** to help
  operations teams prioritize.
- Auto-generate an **operational action plan** (driver coaching, fleet
  follow-up, client recovery response) instead of starting from a blank page.
- Give operations leadership a live **analytics dashboard** and a
  **searchable compliance log** for audits.

## Features

### 🔍 Live Incident Analyzer
Paste any free-text incident report or customer review, pick the vehicle/service
involved, and the engine returns:
- Severity (Low / Medium / High / Critical)
- Category (Driver Behavior, Booking Error, Procedure Missing, Punctuality,
  Customer Experience, Safety, Vehicle Quality)
- Trip value, computed from an editable **cost rate card** (transfer fee or
  hourly at-disposal rate by vehicle type, plus optional extra services)
- Estimated financial impact (EUR), derived from the trip value and severity
- **Client Risk** — every client at this operation is premium/VIP-tier by
  definition, so risk reflects severity plus reputational-escalation signals
  (media, legal, high-profile mentions), not whether a client "is VIP"
- A three-part action plan: **driver actions**, **fleet/operations actions**,
  and a **recommended client response**
- Transparency panel showing which keywords/signals drove the classification
- A **"Save to Audit Log"** button that appends the analyzed incident into
  the Operations Dashboard and Audit & Compliance Log for the rest of the
  session (see note on persistence below)

### 📊 Operations Dashboard
Fleet-wide analytics over a synthetic 50-incident dataset (180-day window):
incident volume by category, severity trend over time, and top drivers by
incident count, plus headline KPIs (total incidents, critical count, average
resolution time, total financial impact).

### 📋 Audit & Compliance Log
A filterable table of the full incident history (category, severity, client
risk, driver, status) with CSV export — the kind of log an operations/quality
manager would review before a client or regulatory audit.

### 💶 Cost Rules (rate card)
An editable table in the sidebar (transfer fee and hourly at-disposal rate
per vehicle type, plus flat fees for extra services like Meet & Greet or
Security Escort) that drives the trip-value and financial-impact estimates.
Defaults live in `data/cost_rules.csv`; edits made in the sidebar apply for
the current session only.

### Note on data persistence
Streamlit Community Cloud has no persistent disk, so incidents saved via
"Save to Audit Log" (and cost-rule edits) live only for the current browser
session and reset when the app restarts — by design, this is a live demo
sandbox, not a production database. The synthetic `data/incidents.csv`
dataset itself is unaffected and always reloads on restart.

## How the AI engine works

The classification engine has two modes:

1. **Rule-based engine (default, no API key required)** — a deterministic
   keyword/regex scoring system (`incident_engine.py`) that matches the
   incident text against category and severity keyword banks, detects VIP
   signals, and computes a financial impact estimate from a
   severity × category multiplier table. This is what runs the demo out of
   the box.
2. **LLM engine (optional)** — if you supply an OpenAI API key in the
   sidebar, the app switches to an LLM-backed classifier
   (`analyze_incident_llm`) that returns the same structured schema via a
   JSON-constrained prompt. If the call fails for any reason, the app
   silently falls back to the rule-based engine so the demo never breaks.

This design means the project **runs instantly for anyone who clones it**,
while still showcasing how an LLM integration would slot into the same
architecture.

## Tech stack

- **Streamlit** — UI framework, dark/luxury custom theme
- **Pandas** — data handling for the incident dataset and audit log
- **Plotly** — interactive analytics charts
- **OpenAI SDK** (optional) — LLM-backed incident classification

## Project structure

```
vip-transport-incident-qc/
├── app.py                 # Streamlit app: 3 tabs (Analyzer, Dashboard, Audit Log)
├── incident_engine.py      # Classification rules, cost model, action plans
├── generate_data.py         # Generates the synthetic 50-incident dataset
├── data/incidents.csv       # Pre-generated demo dataset (deterministic, seeded)
├── data/cost_rules.csv      # Default rate card (vehicle transfer/disposal + extras)
├── requirements.txt
└── .streamlit/config.toml   # Dark/luxury theme
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API key is needed — the
rule-based engine runs the full demo.

To try the optional LLM mode, toggle **"Use LLM (OpenAI) analysis"** in the
sidebar and paste an OpenAI API key (session-only, never stored or logged).

### Regenerating the demo dataset

```bash
python generate_data.py
```

This overwrites `data/incidents.csv` with a fresh (but deterministic, seeded)
synthetic dataset of 50 incidents.

## Disclaimer

All data in this repository — drivers, clients, vehicles, incidents — is
**synthetically generated** for demonstration purposes only and does not
represent any real company, employee, or client.

## Roadmap ideas

- Persist analyzed incidents back into the audit log
- Multi-language incident classification (French/English/Italian keyword banks)
- Driver-level scorecards and trend alerts
- Slack/email notification hook for Critical-severity incidents

## License

MIT
