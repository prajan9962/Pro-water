import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


DATA_FILE = "mettur_synthetic_dataset_2024_2025.csv"
THEME_BLUE = "#0A4FAF"
THEME_ORANGE = "#F28C28"
RISK_MAP = {"low": 0, "medium": 1, "high": 2}
RISK_INV_MAP = {v: k for k, v in RISK_MAP.items()}


st.set_page_config(
    page_title="Smart Water Management & Flood Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --brand-blue: {THEME_BLUE};
            --brand-orange: {THEME_ORANGE};
            --ink: #0f2747;
            --muted: #5e6f86;
            --panel: rgba(255, 255, 255, 0.88);
            --line: rgba(10, 79, 175, 0.15);
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 10%, rgba(10,79,175,0.18), transparent 30%),
                radial-gradient(circle at 90% 8%, rgba(242,140,40,0.18), transparent 28%),
                linear-gradient(130deg, #f4f8ff 0%, #f8fbff 45%, #fff5ea 100%);
            color: var(--ink);
        }}
        .main .block-container {{
            padding-top: 1.2rem;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0c4aa3 0%, #083a80 70%, #0a2f65 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }}
        [data-testid="stSidebar"] * {{
            color: #f3f7ff !important;
        }}
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stCaption {{
            opacity: 0.95;
        }}
        [data-testid="stSidebar"] .st-emotion-cache-16idsys p {{
            color: #d8e6ff !important;
        }}
        .st-emotion-cache-1r6slb0, .st-emotion-cache-1wmy9hl {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(8, 42, 92, 0.08);
            padding: 0.35rem 0.45rem;
        }}
        div[data-testid="stHorizontalBlock"] > div {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 8px 10px 6px 10px;
            box-shadow: 0 8px 24px rgba(10, 52, 110, 0.08);
        }}
        .stDataFrame, div[data-testid="stTable"] {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(10, 52, 110, 0.06);
        }}
        .metric-card {{
            border-radius: 14px;
            padding: 16px;
            background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(245,250,255,0.88));
            box-shadow: 0 10px 24px rgba(10,79,175,0.14);
            border: 1px solid rgba(10,79,175,0.14);
            border-left: 6px solid var(--brand-blue);
            margin-bottom: 10px;
            animation: kpiRise 700ms ease-out both;
        }}
        .metric-title {{
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 600;
        }}
        .metric-value {{
            color: var(--brand-blue);
            font-size: 1.6rem;
            font-weight: 700;
        }}
        h1, h2, h3 {{
            color: var(--ink);
            letter-spacing: 0.2px;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px;
            border: 1px solid rgba(10,79,175,0.25);
            background: linear-gradient(135deg, #0d58be 0%, #0a4fae 75%);
            color: #ffffff;
            font-weight: 600;
            box-shadow: 0 6px 16px rgba(10,79,175,0.25);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: rgba(242,140,40,0.55);
            background: linear-gradient(135deg, #0f62d2 0%, #0b56be 75%);
        }}
        .stAlert {{
            border-radius: 12px;
            border: 1px solid rgba(10,79,175,0.16);
        }}
        .hero-banner {{
            background:
                linear-gradient(135deg, rgba(10,79,175,0.96) 0%, rgba(11,102,187,0.96) 58%, rgba(242,140,40,0.94) 100%);
            border-radius: 16px;
            padding: 18px 20px;
            margin: 6px 0 14px 0;
            color: #ffffff;
            box-shadow: 0 14px 30px rgba(8, 53, 111, 0.28);
            border: 1px solid rgba(255,255,255,0.2);
            animation: bannerIn 700ms ease-out both;
        }}
        .hero-title {{
            font-size: 1.35rem;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: 0.2px;
        }}
        .hero-sub {{
            font-size: 0.95rem;
            opacity: 0.95;
            margin-bottom: 8px;
        }}
        .hero-tags {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .hero-tag {{
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.32);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        @keyframes kpiRise {{
            from {{
                opacity: 0;
                transform: translateY(10px) scale(0.98);
            }}
            to {{
                opacity: 1;
                transform: translateY(0) scale(1);
            }}
        }}
        @keyframes bannerIn {{
            from {{
                opacity: 0;
                transform: translateY(-8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        .risk-low {{
            color: #198754;
            font-weight: 700;
        }}
        .risk-medium {{
            color: #d39e00;
            font-weight: 700;
        }}
        .risk-high {{
            color: #dc3545;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_dataset() -> None:
    try:
        with open(DATA_FILE, "r", encoding="utf-8"):
            return
    except FileNotFoundError:
        st.error(
            f"Dataset not found: {DATA_FILE}. "
            "Place the Mettur dataset CSV in the project folder and rerun."
        )
        st.stop()


@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    ensure_dataset()
    df = pd.read_csv(DATA_FILE)
    df = df.rename(
        columns={
            "Date": "date",
            "Rainfall_mm": "rainfall_mm",
            "Inflow_cusecs": "inflow_cusecs",
            "Outflow_cusecs": "outflow_cusecs",
            "Water_Level_ft": "water_level_ft",
            "Storage_%": "storage_pct",
            "Flood_Alert": "flood_alert",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["flood_alert"] = df["flood_alert"].astype(int)
    # Derived risk label to keep dashboard color states.
    df["flood_level_category"] = np.where(
        (df["flood_alert"] == 1) | (df["water_level_ft"] >= 95),
        "high",
        np.where(df["water_level_ft"] >= 80, "medium", "low"),
    )
    return df.sort_values("date").reset_index(drop=True)


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["rainfall_mm", "inflow_cusecs", "outflow_cusecs", "water_level_ft", "storage_pct"]]


@st.cache_resource
def train_models(df: pd.DataFrame):
    x = prep_features(df)
    y_class = np.where(df["flood_alert"] == 1, 2, np.where(df["water_level_ft"] >= 80, 1, 0))
    y_harvest = np.maximum(0, df["inflow_cusecs"] * 0.0018 + df["rainfall_mm"] * 0.42 - df["outflow_cusecs"] * 0.00095)
    y_release = np.maximum(
        0,
        (df["water_level_ft"] - 88) * 900
        + (df["inflow_cusecs"] - df["outflow_cusecs"]) * 0.26
        + (df["storage_pct"] - 86) * 450
        + (df["rainfall_mm"] - 20) * 55,
    )
    y_release = np.round(y_release, 1)

    flood_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    flood_model.fit(x, y_class)

    harvest_model = RandomForestRegressor(n_estimators=250, random_state=42)
    harvest_model.fit(x, y_harvest)
    release_model = RandomForestRegressor(n_estimators=240, random_state=42)
    release_model.fit(x, y_release)
    return flood_model, harvest_model, release_model


@st.cache_data(ttl=30)
def cached_predictions(df: pd.DataFrame):
    flood_model, harvest_model, _release_model = train_models(df)
    x = prep_features(df)
    risk_class = flood_model.predict(x)
    risk_prob = flood_model.predict_proba(x)
    max_risk = (risk_prob.max(axis=1) * 100).round(1)
    harvest = harvest_model.predict(x).round(2)
    return risk_class, max_risk, harvest


@st.cache_data(ttl=30)
def forecast_7_days(df: pd.DataFrame) -> pd.DataFrame:
    flood_model, harvest_model, release_model = train_models(df)
    recent = df.tail(14).copy()
    last_row = recent.iloc[-1]
    roll_rain = recent["rainfall_mm"].mean()
    roll_inflow = recent["inflow_cusecs"].mean()
    roll_outflow = recent["outflow_cusecs"].mean()
    roll_level = recent["water_level_ft"].mean()
    roll_storage = recent["storage_pct"].mean()

    future = []
    for i in range(1, 8):
        dt = last_row["date"] + timedelta(days=i)
        seasonal_boost = 1.2 if dt.month in [10, 11, 12] else 1.0
        rainfall = max(0, roll_rain + np.sin(i / 1.8) * 5.5) * seasonal_boost
        inflow = np.clip(roll_inflow + rainfall * 780 + np.sin(i / 2) * 2500, 1000, 120000)
        outflow = np.clip(roll_outflow + rainfall * 520 + np.cos(i / 2.5) * 1900, 1000, 110000)
        water_level = np.clip(roll_level + (inflow - outflow) / 65000 + rainfall * 0.018, 40, 100)
        storage = np.clip(roll_storage + (inflow - outflow) / 70000 + rainfall * 0.016, 40, 100)

        future.append(
            {
                "date": dt,
                "rainfall_mm": round(rainfall, 1),
                "inflow_cusecs": round(float(inflow), 2),
                "outflow_cusecs": round(float(outflow), 2),
                "water_level_ft": round(float(water_level), 2),
                "storage_pct": round(float(storage), 2),
            }
        )

    forecast_df = pd.DataFrame(future)
    x_future = prep_features(forecast_df)
    pred_class = flood_model.predict(x_future)
    pred_prob = flood_model.predict_proba(x_future).max(axis=1)
    pred_harvest = harvest_model.predict(x_future)
    pred_release = release_model.predict(x_future)

    forecast_df["predicted_flood_level"] = [RISK_INV_MAP[int(v)] for v in pred_class]
    forecast_df["risk_score_0_100"] = (pred_prob * 100).round(1)
    forecast_df["harvest_potential_ml"] = pred_harvest.round(2)
    forecast_df["recommended_release_cusecs"] = np.clip(pred_release.round(1), 0, None)
    forecast_df["release_required"] = np.where(forecast_df["recommended_release_cusecs"] >= 30, "Yes", "No")
    return forecast_df


@st.cache_data(ttl=30)
def release_decision_now(df: pd.DataFrame):
    _flood_model, _harvest_model, release_model = train_models(df)
    latest = df.tail(1).copy()
    latest_x = prep_features(latest)
    predicted_release = float(np.clip(release_model.predict(latest_x)[0], 0, None))
    release_required = predicted_release >= 30
    return release_required, round(predicted_release, 1), latest.iloc[0]


def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Smart Water Management & Flood Prediction System</div>
            <div class="hero-sub">Mettur Dam Authority - AI-enabled offline decision dashboard</div>
            <div class="hero-tags">
                <span class="hero-tag">100% Offline</span>
                <span class="hero-tag">AI Forecasting</span>
                <span class="hero-tag">Dam Release Intelligence</span>
                <span class="hero-tag">30s Auto Refresh</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label_html(label: str) -> str:
    css = {"low": "risk-low", "medium": "risk-medium", "high": "risk-high"}.get(label, "risk-medium")
    return f"<span class='{css}'>{label.upper()}</span>"


def generate_recommendations(df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    avg_rain = df["rainfall_mm"].mean()
    avg_outflow = df["outflow_cusecs"].mean()
    high_days = (df["flood_alert"] == 1).sum()
    wet_month_ratio = (df[df["date"].dt.month.isin([10, 11, 12])]["rainfall_mm"].mean() / max(avg_rain, 1))
    future_high_days = (forecast_df["predicted_flood_level"] == "high").sum()

    recommendations = [
        (
            "Increase pre-monsoon controlled release scheduling",
            "Creates storage buffer before high inflow days",
            220.0,
            362.0,
        ),
        (
            "Automate Mettur gate opening with forecast thresholds",
            "Reduces emergency operations and downstream surge",
            145.0,
            285.0,
        ),
        (
            "Upgrade inflow-side telemetry and gauge stations",
            "Improves inflow forecasting and gate timing accuracy",
            180.0,
            318.0,
        ),
        (
            "Strengthen canal release corridors downstream",
            "Supports safer high-volume outflow distribution",
            95.0,
            196.0,
        ),
        (
            "Schedule desilting before northeast monsoon",
            "Raises effective flow capacity under peak rainfall",
            110.0,
            214.0,
        ),
    ]

    rec_df = pd.DataFrame(recommendations, columns=["action", "impact", "investment_lakh", "benefit_lakh"])
    rec_df["roi_pct"] = ((rec_df["benefit_lakh"] - rec_df["investment_lakh"]) / rec_df["investment_lakh"] * 100).round(1)
    rec_df["priority_score"] = (
        rec_df["roi_pct"] * 0.5
        + wet_month_ratio * 10
        + high_days * 0.7
        + future_high_days * 3
        + (avg_outflow / 8000)
    ).round(1)
    return rec_df.sort_values("priority_score", ascending=False).head(5).reset_index(drop=True)


def build_basic_pdf_bytes(title: str, lines: list[str]) -> bytes:
    clean_lines = [line.replace("\\", "/").replace("(", "[").replace(")", "]") for line in lines]
    y = 770
    text_lines = [f"BT /F1 18 Tf 50 {y} Td ({title}) Tj ET"]
    y -= 30
    for line in clean_lines:
        text_lines.append(f"BT /F1 11 Tf 50 {y} Td ({line}) Tj ET")
        y -= 18
        if y < 80:
            break
    content_stream = "\n".join(text_lines).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj"
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(b"5 0 obj << /Length " + str(len(content_stream)).encode() + b" >> stream\n" + content_stream + b"\nendstream endobj")

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    xref_positions = [0]
    for obj in objects:
        xref_positions.append(pdf.tell())
        pdf.write(obj + b"\n")
    xref_start = pdf.tell()
    pdf.write(f"xref\n0 {len(xref_positions)}\n".encode())
    pdf.write(b"0000000000 65535 f \n")
    for pos in xref_positions[1:]:
        pdf.write(f"{pos:010d} 00000 n \n".encode())
    pdf.write(f"trailer << /Size {len(xref_positions)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode())
    return pdf.getvalue()


def page_home(df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.title("Smart Water Management & Flood Prediction System")
    st.caption("Mettur Dam Decision Dashboard - Offline Intelligence Suite")

    recent = df.tail(30)
    wastage_potential = np.maximum(0, recent["inflow_cusecs"] - recent["outflow_cusecs"]).sum() * 0.0007
    flood_risk_score = int(round(forecast_df["risk_score_0_100"].mean()))
    current_outflow = recent["outflow_cusecs"].mean()
    optimized = current_outflow * 0.9
    savings = max(0, current_outflow - optimized) * 30

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Excess Storage Pressure (last 30 days)", f"{wastage_potential:,.1f} ML eq.")
    with c2:
        metric_card("Flood Risk Score (0-100)", f"{flood_risk_score}")
    with c3:
        metric_card("Outflow Optimization Savings (monthly)", f"{savings:,.0f} cusecs")

    line_fig = px.line(
        df,
        x="date",
        y=["rainfall_mm", "water_level_ft", "outflow_cusecs"],
        title="Mettur Core Dynamics Timeline",
        color_discrete_sequence=[THEME_BLUE, THEME_ORANGE, "#2D9CDB"],
    )
    line_fig.update_layout(legend_title_text="", height=410, margin=dict(l=20, r=20, t=48, b=20))
    st.plotly_chart(line_fig, use_container_width=True)


def page_flood_predictor(df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.header("Flood Predictor")
    st.write("7-day AI forecast driven by offline RandomForest model.")

    show = forecast_df[
        [
            "date",
            "rainfall_mm",
            "water_level_ft",
            "outflow_cusecs",
            "predicted_flood_level",
            "risk_score_0_100",
            "release_required",
            "recommended_release_cusecs",
        ]
    ].copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, use_container_width=True, hide_index=True)

    color_map = {"low": "#1A9B5F", "medium": "#F2B134", "high": "#E34F4F"}
    fig = px.bar(
        forecast_df,
        x="date",
        y="risk_score_0_100",
        color="predicted_flood_level",
        title="7-Day Flood Risk Outlook",
        color_discrete_map=color_map,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(fig, use_container_width=True)

    risk_now = forecast_df.iloc[0]["predicted_flood_level"]
    st.markdown(f"Current forecast risk level: {risk_label_html(risk_now)}", unsafe_allow_html=True)


def page_wastage_harvester(df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.header("Wastage Harvester")
    rain_months = df[df["date"].dt.month.isin([10, 11, 12])]
    rain_months = rain_months.assign(
        wastage_ml=(np.maximum(0, rain_months["inflow_cusecs"] - rain_months["outflow_cusecs"]) * 0.001).round(2)
    )

    fig = px.area(
        rain_months,
        x="date",
        y="wastage_ml",
        title="Rainy Season Water Wastage Potential",
        color_discrete_sequence=[THEME_ORANGE],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

    avg_storage = df["storage_pct"].mean()
    target_storage = min(100.0, avg_storage + 4.0)
    st.info(
        f"Storage recommendation: keep effective storage around {target_storage:.2f}% "
        f"(current avg {avg_storage:.2f}%) with pre-release before peak inflow."
    )

    harvest_total = forecast_df["harvest_potential_ml"].sum()
    st.success(f"Predicted harvest potential for next 7 days: {harvest_total:,.2f} ML")


def page_dam_optimizer(df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.header("Dam Optimizer")
    recent = df.tail(21).copy()
    recent["recommended_discharge"] = (
        recent["outflow_cusecs"] + np.maximum(0, recent["inflow_cusecs"] - recent["outflow_cusecs"]) * 0.25
    ).clip(lower=1000)
    recent["savings_cusecs"] = (recent["recommended_discharge"] - recent["outflow_cusecs"]).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["outflow_cusecs"],
            mode="lines+markers",
            name="Current Outflow",
            line=dict(color=THEME_BLUE, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["recommended_discharge"],
            mode="lines+markers",
            name="AI Recommended",
            line=dict(color=THEME_ORANGE, width=3),
        )
    )
    fig.update_layout(title="Current vs Optimized Outflow", height=390, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    alloc = pd.DataFrame({"Sector": ["Agri", "Lakes", "Industry"], "Share": [40, 40, 20]})
    pie = px.pie(
        alloc,
        names="Sector",
        values="Share",
        hole=0.52,
        title="AI Water Allocation",
        color="Sector",
        color_discrete_map={"Agri": "#2D9CDB", "Lakes": THEME_BLUE, "Industry": THEME_ORANGE},
    )
    pie.update_layout(height=360, margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(pie, use_container_width=True)

    st.subheader("Gate Control Decision (Next 7 Days)")
    overflow_threshold = st.slider(
        "Overflow release threshold (cusecs)",
        min_value=10,
        max_value=120,
        value=30,
        step=1,
        help="If the model-recommended release exceeds this threshold, the gates are predicted to be opened.",
    )
    cusec_to_m3_per_hr = 101.942507  # 1 cusec = 0.028316846592 m^3/s => *3600

    gate_plan = forecast_df.copy()
    gate_plan["gate_state"] = np.where(gate_plan["recommended_release_cusecs"] >= overflow_threshold, "OPEN", "CLOSED")
    gate_plan["release_m3_per_hr"] = (gate_plan["recommended_release_cusecs"].astype(float) * cusec_to_m3_per_hr).round(1)
    overall_open = (gate_plan["gate_state"] == "OPEN").any()
    worst_idx = gate_plan["recommended_release_cusecs"].astype(float).idxmax()
    worst_day = gate_plan.loc[worst_idx]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Overall Gate Status (Next 7 Days)", "OPEN" if overall_open else "CLOSED")
    with c2:
        metric_card("Worst Day Release (cusecs)", f"{float(worst_day['recommended_release_cusecs']):,.1f}")
    with c3:
        metric_card("Worst Day Release (m³/hour)", f"{float(worst_day['release_m3_per_hr']):,.1f}")

    st.subheader("Recommended Release for Next 7 Days")
    rel_plot_df = gate_plan[["date", "recommended_release_cusecs", "gate_state", "release_m3_per_hr"]].copy()
    rel_plot_df["decision_color"] = np.where(rel_plot_df["gate_state"] == "OPEN", "Release", "Hold")
    release_fig = px.bar(
        rel_plot_df,
        x="date",
        y="recommended_release_cusecs",
        color="decision_color",
        title="AI Release Plan (7-Day Outlook)",
        color_discrete_map={"Release": "#E34F4F", "Hold": "#1A9B5F"},
        labels={"recommended_release_cusecs": "Recommended release (cusecs)", "date": "Date", "decision_color": "Decision"},
    )
    release_fig.add_hline(
        y=overflow_threshold,
        line_dash="dash",
        line_color=THEME_ORANGE,
        annotation_text=f"Threshold = {overflow_threshold} cusecs",
    )
    release_fig.update_layout(height=360, margin=dict(l=10, r=10, t=52, b=10))
    st.plotly_chart(release_fig, use_container_width=True)

    st.subheader("Gate Status Timeline (Next 7 Days)")
    timeline_df = gate_plan[["date", "gate_state"]].copy()
    timeline_df["date"] = timeline_df["date"].dt.strftime("%Y-%m-%d")
    timeline_df["gate_numeric"] = np.where(timeline_df["gate_state"] == "OPEN", 1, 0)
    timeline_fig = px.line(
        timeline_df,
        x="date",
        y="gate_numeric",
        markers=True,
        color="gate_state",
        color_discrete_map={"OPEN": "#E34F4F", "CLOSED": "#1A9B5F"},
        title="Dam Gate Control Timeline (1=OPEN, 0=CLOSED)",
        labels={"gate_numeric": "Gate state", "date": "Date", "gate_state": "State"},
    )
    timeline_fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["CLOSED", "OPEN"], range=[-0.1, 1.1])
    timeline_fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(timeline_fig, use_container_width=True)

    st.subheader("Daily Hourly Release Plan")
    table = gate_plan[["date", "gate_state", "recommended_release_cusecs", "release_m3_per_hr"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table = table.rename(
        columns={
            "recommended_release_cusecs": "Recommended release (cusecs)",
            "release_m3_per_hr": "Release per hour (m³/hour)",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Optional: show "now" decision (latest row) as additional context for the demo.
    release_required_now, predicted_release_now, latest_row = release_decision_now(df)
    # Re-apply the same overflow threshold used for the next-7-days plan.
    release_required_now = float(predicted_release_now) >= overflow_threshold
    decision_text_now = "OPEN" if release_required_now else "CLOSED"
    if release_required_now:
        st.warning(f"Now (latest conditions): Gates should be {decision_text_now}. Recommended: {predicted_release_now:,.1f} cusecs.")
    else:
        st.info(f"Now (latest conditions): Gates should be {decision_text_now}. Suggested: {predicted_release_now:,.1f} cusecs.")

    st.subheader("Savings Calculator")
    c1, c2 = st.columns(2)
    with c1:
        current = st.number_input("Current average outflow (cusecs)", min_value=500.0, value=float(recent["outflow_cusecs"].mean()))
    with c2:
        reduction_pct = st.slider("Optimization reduction (%)", 1, 30, 12)
    monthly_savings = current * (reduction_pct / 100) * 30
    st.success(f"Estimated monthly savings: {monthly_savings:,.1f} cusecs")

    st.subheader("Manual Release What-If (Overflow Risk Impact)")
    m1, m2 = st.columns(2)
    with m1:
        trial_release = st.slider(
            "Trial release amount (cusecs)",
            min_value=0,
            max_value=500,
            value=int(round(predicted_release_now)),
            step=5,
        )
    with m2:
        baseline_risk = float(
            np.clip(
                (latest_row["water_level_ft"] * 0.8)
                + (latest_row["rainfall_mm"] * 0.7)
                + ((latest_row["inflow_cusecs"] - latest_row["outflow_cusecs"]) / 2500),
                0,
                100,
            )
        )
        expected_risk_after = float(np.clip(baseline_risk - trial_release * 0.12, 0, 100))
        st.metric("Estimated overflow risk after release", f"{expected_risk_after:.1f}/100", f"-{baseline_risk - expected_risk_after:.1f}")


def page_data_explorer(df: pd.DataFrame):
    st.header("Data Explorer")
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    start, end = st.date_input("Filter date range", (min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(start, tuple):
        start, end = start
    filtered = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

    metric = st.selectbox(
        "Select metric",
        ["rainfall_mm", "inflow_cusecs", "outflow_cusecs", "water_level_ft", "storage_pct", "flood_alert"],
    )
    ts = px.line(filtered, x="date", y=metric, title=f"{metric} time series", color_discrete_sequence=[THEME_BLUE])
    ts.update_layout(height=350, margin=dict(l=10, r=10, t=46, b=10))
    st.plotly_chart(ts, use_container_width=True)

    corr_cols = ["rainfall_mm", "inflow_cusecs", "outflow_cusecs", "water_level_ft", "storage_pct", "flood_alert"]
    corr = filtered[corr_cols].corr().round(2)
    heat = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        color_continuous_scale=[[0, "#fff5e6"], [0.5, "#7db7ff"], [1, THEME_BLUE]],
        aspect="auto",
    )
    heat.update_layout(height=420, margin=dict(l=10, r=10, t=46, b=10))
    st.plotly_chart(heat, use_container_width=True)

    st.dataframe(filtered, use_container_width=True, hide_index=True)


def page_recommendations(df: pd.DataFrame, forecast_df: pd.DataFrame):
    st.header("Recommendations")
    rec_df = generate_recommendations(df, forecast_df)

    st.subheader("Top 5 Actionable Suggestions")
    display_df = rec_df[["action", "impact", "investment_lakh", "benefit_lakh", "roi_pct"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    roi_fig = px.bar(
        rec_df,
        x="action",
        y="roi_pct",
        title="Infrastructure Expansion ROI",
        color="roi_pct",
        color_continuous_scale=[[0, "#ffcf99"], [1, THEME_BLUE]],
    )
    roi_fig.update_layout(height=390, xaxis_title="", yaxis_title="ROI (%)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(roi_fig, use_container_width=True)

    pdf_lines = [
        "Smart Water Management - Priority Actions",
        "",
    ]
    for idx, row in rec_df.iterrows():
        pdf_lines.append(
            f"{idx + 1}. {row['action']} | ROI: {row['roi_pct']}% | Invest: {row['investment_lakh']}L | Benefit: {row['benefit_lakh']}L"
        )
    pdf_bytes = build_basic_pdf_bytes("Mettur Dam Recommendations", pdf_lines)
    st.download_button(
        "Export Recommendations as PDF",
        data=pdf_bytes,
        file_name="mettur_dam_recommendations.pdf",
        mime="application/pdf",
    )


def main():
    inject_theme()
    auto_refresh = st.sidebar.toggle("Auto refresh every 30s", value=True)
    df = load_data()
    _risk_class, _risk_score, _harvest = cached_predictions(df)
    forecast_df = forecast_7_days(df)
    render_hero_banner()

    st.sidebar.markdown("### Mettur Dam Authority")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 HOME", "🌧️ FLOOD PREDICTOR", "💧 WASTAGE HARVESTER", "🚰 DAM OPTIMIZER", "📊 DATA EXPLORER", "⚙️ RECOMMENDATIONS"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Offline mode: active")
    st.sidebar.caption(f"Dataset rows: {len(df)}")
    st.sidebar.caption("Refresh cycle: 30 seconds")

    if page == "🏠 HOME":
        page_home(df, forecast_df)
    elif page == "🌧️ FLOOD PREDICTOR":
        page_flood_predictor(df, forecast_df)
    elif page == "💧 WASTAGE HARVESTER":
        page_wastage_harvester(df, forecast_df)
    elif page == "🚰 DAM OPTIMIZER":
        page_dam_optimizer(df, forecast_df)
    elif page == "📊 DATA EXPLORER":
        page_data_explorer(df)
    else:
        page_recommendations(df, forecast_df)

    if auto_refresh:
        components.html(
            """
            <script>
            setTimeout(function() {
                window.parent.location.reload();
            }, 30000);
            </script>
            """,
            height=0,
            width=0,
        )


if __name__ == "__main__":
    main()
