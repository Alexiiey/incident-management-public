"""VIP Transport - AI Incident & Quality Control System.

Streamlit application demonstrating an end-to-end quality control workflow
for a luxury chauffeur (NCC) operation: live incident triage, an operations
analytics dashboard, and an auditable compliance log.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from incident_engine import VEHICLE_TYPES, IncidentAnalysis, analyze_incident

DATA_PATH = Path(__file__).parent / "data" / "incidents.csv"
COST_RULES_PATH = Path(__file__).parent / "data" / "cost_rules.csv"

SEVERITY_COLORS = {
    "Critical": "#E4572E",
    "High": "#F2A541",
    "Medium": "#C9A227",
    "Low": "#4C9A6A",
}
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

LOG_COLUMNS = [
    "incident_id", "date", "driver_name", "vehicle_type", "service_type",
    "disposal_hours", "client_tier", "category", "severity", "description",
    "trip_value_eur", "financial_impact_eur", "client_risk", "resolved",
    "resolution_time_hours", "auditor",
]

st.set_page_config(
    page_title="VIP Transport - AI Incident & Quality Control",
    page_icon="🚘",
    layout="wide",
)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .kpi-card {
            background: linear-gradient(160deg, #1A1A1D 0%, #131315 100%);
            border: 1px solid #2A2A2E;
            border-left: 4px solid var(--accent, #C9A227);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            height: 100%;
        }
        .kpi-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9A9AA2;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #F2F2F2;
        }
        .action-box {
            background: #151517;
            border: 1px solid #2A2A2E;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            height: 100%;
        }
        .action-box h4 {
            margin-top: 0;
            color: #C9A227;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, accent: str = "#C9A227") -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


def action_box(title: str, items: list[str]) -> str:
    bullets = "".join(f"<li>{item}</li>" for item in items)
    return f'<div class="action-box"><h4>{title}</h4><ul>{bullets}</ul></div>'


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data
def load_incidents() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@st.cache_data
def load_default_cost_rules() -> pd.DataFrame:
    return pd.read_csv(COST_RULES_PATH)


def get_combined_incidents(base_df: pd.DataFrame) -> pd.DataFrame:
    """Base synthetic dataset + any incidents saved from the live analyzer
    during this session."""
    saved = st.session_state.get("saved_incidents", [])
    if saved:
        saved_df = pd.DataFrame(saved)
        combined = pd.concat([saved_df, base_df], ignore_index=True, sort=False)
    else:
        combined = base_df.copy()
    combined["severity"] = pd.Categorical(combined["severity"], categories=SEVERITY_ORDER, ordered=True)
    return combined


# --------------------------------------------------------------------------
# Sidebar - AI engine & cost rules configuration
# --------------------------------------------------------------------------
def render_sidebar() -> tuple[bool, str | None, pd.DataFrame]:
    st.sidebar.header("⚙️ Analysis Engine")
    st.sidebar.caption(
        "The rule-based engine works instantly with no API key. Optionally "
        "supply an OpenAI key to switch to LLM-powered analysis."
    )
    use_llm = st.sidebar.toggle("Use LLM (OpenAI) analysis", value=False)
    api_key = None
    if use_llm:
        api_key = st.sidebar.text_input("OpenAI API key", type="password")
        if not api_key:
            st.sidebar.warning("No API key provided — falling back to the rule-based engine.")

    st.sidebar.divider()
    st.sidebar.subheader("💶 Cost Rules (rate card)")
    st.sidebar.caption(
        "Transfer/at-disposal rates and additional service fees used to "
        "estimate the value of the affected trip. Edit freely for this session."
    )
    if "cost_rules" not in st.session_state:
        st.session_state["cost_rules"] = load_default_cost_rules().copy()
    edited_rules = st.sidebar.data_editor(
        st.session_state["cost_rules"],
        num_rows="dynamic",
        width="stretch",
        key="cost_rules_editor",
        hide_index=True,
    )
    st.session_state["cost_rules"] = edited_rules

    st.sidebar.divider()
    st.sidebar.caption(
        "**VIP Transport QC** demonstrates an operations quality-control "
        "workflow: incident triage, fleet analytics, and compliance audit "
        "logging for a luxury chauffeur (NCC) business."
    )
    return use_llm, api_key, edited_rules


# --------------------------------------------------------------------------
# Tab 1 - Live Incident Analyzer
# --------------------------------------------------------------------------
def render_incident_analyzer(use_llm: bool, api_key: str | None, cost_rules_df: pd.DataFrame) -> None:
    st.subheader("Live Incident Analyzer")
    st.caption(
        "Paste a raw incident report, driver note, or customer review below. "
        "The engine classifies severity, category, financial exposure, and "
        "client risk, then drafts an operational action plan."
    )

    example = (
        "Repeat client noted: The driver was 25 minutes late for the airport "
        "pickup and was noticeably rude when the client raised the issue."
    )

    if "incident_text" not in st.session_state:
        st.session_state["incident_text"] = ""

    if st.button("Load example"):
        st.session_state["incident_text"] = example
        st.rerun()

    text = st.text_area(
        "Incident report / customer review",
        key="incident_text",
        height=140,
        placeholder=example,
    )

    with st.expander("Trip & cost details", expanded=True):
        c1, c2, c3 = st.columns(3)
        vehicle_options = sorted(cost_rules_df.loc[cost_rules_df["service_type"] == "Transfer", "item"].unique())
        vehicle = c1.selectbox("Vehicle type", vehicle_options or VEHICLE_TYPES)
        service_type = c2.selectbox("Service type", ["Transfer", "At Disposal (hourly)"])
        disposal_hours = None
        if service_type == "At Disposal (hourly)":
            disposal_hours = c3.number_input("Hours at disposal", min_value=1, value=3, step=1)
        extra_options = sorted(cost_rules_df.loc[cost_rules_df["service_type"] == "Other Service", "item"].unique())
        extra_services = st.multiselect("Additional services involved", extra_options)

        d1, d2 = st.columns(2)
        driver_name = d1.text_input("Driver / supplier name (optional)")
        auditor_name = d2.text_input("Auditor name (optional)", value="Live Analysis")

    if st.button("Analyze Incident", type="primary"):
        if not text.strip():
            st.warning("Please paste an incident report before analyzing.")
        else:
            result = analyze_incident(
                text,
                vehicle=vehicle,
                service_type=service_type,
                disposal_hours=disposal_hours,
                extra_services=extra_services,
                cost_rules_df=cost_rules_df,
                use_llm=use_llm,
                api_key=api_key,
            )
            st.session_state["last_result"] = result
            st.session_state["last_context"] = {
                "text": text,
                "vehicle": vehicle,
                "service_type": service_type,
                "disposal_hours": disposal_hours,
                "driver_name": driver_name,
                "auditor_name": auditor_name,
            }

    result: IncidentAnalysis | None = st.session_state.get("last_result")
    if result is None:
        return

    ctx = st.session_state.get("last_context", {})
    st.caption(f"Engine used: **{result.engine_used}** · Confidence: **{result.confidence}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Severity", result.severity, SEVERITY_COLORS.get(result.severity, "#C9A227")), unsafe_allow_html=True)
    c2.markdown(kpi_card("Category", result.category), unsafe_allow_html=True)
    c3.markdown(kpi_card("Trip Value", f"€{result.trip_value_eur:,.0f}"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Est. Financial Impact", f"€{result.financial_impact_eur:,.0f}"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Client Risk", result.client_risk, SEVERITY_COLORS.get(result.client_risk, "#C9A227")), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Operational Action Plan")
    a1, a2, a3 = st.columns(3)
    a1.markdown(action_box("Driver Actions", result.driver_actions), unsafe_allow_html=True)
    a2.markdown(action_box("Fleet / Operations Actions", result.fleet_actions), unsafe_allow_html=True)
    a3.markdown(action_box("Client Response", [result.client_response]), unsafe_allow_html=True)

    if result.matched_keywords:
        with st.expander("Why this classification? (matched signals)"):
            st.write(", ".join(f"`{kw}`" for kw in result.matched_keywords))

    st.write("")
    if st.button("💾 Save to Audit Log"):
        row = {
            "incident_id": f"LIVE-{datetime.now():%Y%m%d%H%M%S}",
            "date": pd.Timestamp.now(),
            "driver_name": ctx.get("driver_name") or "Unknown",
            "vehicle_type": ctx.get("vehicle"),
            "service_type": ctx.get("service_type"),
            "disposal_hours": ctx.get("disposal_hours"),
            "client_tier": "VIP",
            "category": result.category,
            "severity": result.severity,
            "description": ctx.get("text", ""),
            "trip_value_eur": result.trip_value_eur,
            "financial_impact_eur": result.financial_impact_eur,
            "client_risk": result.client_risk,
            "resolved": False,
            "resolution_time_hours": None,
            "auditor": ctx.get("auditor_name") or "Live Analysis",
        }
        st.session_state.setdefault("saved_incidents", []).append(row)
        st.success("Saved — check the Operations Dashboard and Audit & Compliance Log tabs.")


# --------------------------------------------------------------------------
# Tab 2 - Operations Dashboard
# --------------------------------------------------------------------------
def render_operations_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Operations Dashboard")
    st.caption(f"Fleet-wide view of {len(df)} incidents (synthetic demo dataset + any incidents saved this session).")

    total = len(df)
    critical = int((df["severity"] == "Critical").sum())
    avg_resolution = df["resolution_time_hours"].dropna().mean()
    total_impact = df["financial_impact_eur"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Incidents", f"{total}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Critical Incidents", f"{critical}", SEVERITY_COLORS["Critical"]), unsafe_allow_html=True)
    c3.markdown(kpi_card("Avg Resolution Time", f"{avg_resolution:.1f} h" if pd.notna(avg_resolution) else "N/A"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Total Financial Impact", f"€{total_impact:,.0f}"), unsafe_allow_html=True)

    st.write("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Incidents by Category**")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Incidents"]
        fig_cat = px.bar(
            cat_counts.sort_values("Incidents"),
            x="Incidents", y="Category", orientation="h",
            color="Incidents", color_continuous_scale=["#3A3A3E", "#C9A227"],
        )
        fig_cat.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_cat, width="stretch")

    with col_right:
        st.markdown("**Severity Trend Over Time**")
        trend = df.copy()
        trend["week"] = trend["date"].dt.to_period("W").dt.start_time
        trend_counts = trend.groupby(["week", "severity"], observed=True).size().reset_index(name="count")
        fig_trend = px.area(
            trend_counts, x="week", y="count", color="severity",
            category_orders={"severity": SEVERITY_ORDER},
            color_discrete_map=SEVERITY_COLORS,
        )
        fig_trend.update_layout(legend_title_text="Severity")
        st.plotly_chart(fig_trend, width="stretch")

    st.markdown("**Top Drivers by Incident Count**")
    top_drivers = df["driver_name"].value_counts().head(5).reset_index()
    top_drivers.columns = ["Driver", "Incidents"]
    fig_drivers = px.bar(
        top_drivers.sort_values("Incidents"),
        x="Incidents", y="Driver", orientation="h",
        color="Incidents", color_continuous_scale=["#3A3A3E", "#E4572E"],
    )
    fig_drivers.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
    st.plotly_chart(fig_drivers, width="stretch")


# --------------------------------------------------------------------------
# Tab 3 - Audit & Compliance Log
# --------------------------------------------------------------------------
def render_audit_log(df: pd.DataFrame) -> None:
    st.subheader("Audit & Compliance Log")
    st.caption("Filterable history of quality control checks and incident audits.")

    f1, f2, f3, f4, f5 = st.columns(5)
    categories = f1.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    severities = f2.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    client_risks = f3.multiselect("Client Risk", SEVERITY_ORDER, default=SEVERITY_ORDER)
    drivers = f4.multiselect("Driver", sorted(df["driver_name"].dropna().unique()), default=sorted(df["driver_name"].dropna().unique()))
    resolved_filter = f5.selectbox("Status", ["All", "Resolved", "Open"])

    filtered = df[
        df["category"].isin(categories)
        & df["severity"].isin(severities)
        & df["client_risk"].isin(client_risks)
        & df["driver_name"].isin(drivers)
    ]
    if resolved_filter == "Resolved":
        filtered = filtered[filtered["resolved"]]
    elif resolved_filter == "Open":
        filtered = filtered[~filtered["resolved"]]

    st.dataframe(
        filtered.sort_values("date", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download filtered log (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="audit_compliance_log.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    inject_css()
    use_llm, api_key, cost_rules_df = render_sidebar()

    st.title("🚘 VIP Transport — AI Incident & Quality Control System")
    st.caption("Operations quality control, incident triage and compliance auditing for luxury chauffeur services.")

    tab1, tab2, tab3 = st.tabs(["🔍 Live Incident Analyzer", "📊 Operations Dashboard", "📋 Audit & Compliance Log"])

    with tab1:
        render_incident_analyzer(use_llm, api_key, cost_rules_df)

    combined_df = get_combined_incidents(load_incidents())
    with tab2:
        render_operations_dashboard(combined_df)
    with tab3:
        render_audit_log(combined_df)


if __name__ == "__main__":
    main()
