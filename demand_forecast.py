# -------------------------------
# Core libraries
# -------------------------------
import streamlit as st
import pandas as pd

# -------------------------------
# Modelling
# -------------------------------
from prophet import Prophet

# -------------------------------
# Visualisation
# -------------------------------
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
# -------------------------------
# Utilities
# -------------------------------
from io import StringIO
import markdown
import textwrap

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------


st.set_page_config(
    page_title="Demand Forecast",
    layout="wide"
)

st.title("📈 Demand Forecast")
st.markdown(
    """
        
    This tool estimates **future monthly demand** using historical data and the **Facebook / Meta Prophet** model.  
    It is intended to support **capacity planning, workforce discussions, and risk conversations**.

    The forecast describes a **likely range of demand**, not a precise prediction.
    Observed activity will always vary month‑to‑month due to operational and external factors.

    Existing backlog is **not** included in the forecast.

    """
)

# --------------------------------------------------
# USER INSTRUCTIONS
# --------------------------------------------------

st.subheader("📂 Upload your data")

st.info("""
Upload a CSV file containing **monthly activity data**.

Your file should include at least:
- A **month column** in **DD/MM/YYYY format**
- An **activity column** (numeric values)

Example:

| Month      | Activity |
|------------|----------|
| 01/01/2023 | 1200     |
| 01/02/2023 | 1150     |
| 01/03/2023 | 1300     |

For a more reliable forecast, you should include **at least** 2 years' worth of data.
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload monthly activity CSV (DD/MM/YYYY format)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# --------------------------------------------------
# DATA LOAD + COLUMN SELECTION
# --------------------------------------------------
df_raw = pd.read_csv(uploaded_file)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔍 Data load")
    st.dataframe(df_raw.head(), width='stretch')
    #st.write("📂 Columns detected:", list(df_raw.columns))

with col_right:
    st.subheader("🧱 Column selection")

    DATE_COLUMN = st.selectbox(
        "Select the date column",
        df_raw.columns
    )

    default_index = 1 if len(df_raw.columns) > 1 else 0
    VALUE_COLUMN = st.selectbox(
        "Select the demand count column",
        df_raw.columns,
        index=default_index
    )

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------
df = df_raw.copy()

df["ds"] = pd.to_datetime(
    df[DATE_COLUMN],
    format="%d/%m/%Y",
    dayfirst=True
)

df["y"] = df[VALUE_COLUMN]

df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)

# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------
st.subheader("✅ Data validation")

with st.expander("❓How to interpret the data checks"):
    st.caption(
        """
        The forecast is based solely on the historical data provided.
        The longer and more consistent the time series, the more reliable the estimated trend
        and seasonal patterns will be.

        As a rule of thumb:
        - **18+ months** supports stable trend and seasonality estimates
        - Shorter histories should be used for **exploratory planning only**
        """
    )

st.write(f"**Date range:** {df['ds'].min().date()} → {df['ds'].max().date()}")
st.write(f"**Number of months:** {len(df)}")

if len(df) < 18:
    st.warning("Less than 18 months of data – forecast reliability may be limited.")

# --------------------------------------------------
# SIDEBAR – MODEL SETTINGS
# --------------------------------------------------
st.sidebar.header("⚙️ Model settings")

st.sidebar.caption(
    """
    These controls allow you to align the forecast with **service knowledge**.

    There is no single 'correct' setting.
    Adjustments should reflect genuine understanding of how demands behave,
    not optimisation for the most visually appealing result.
        
    """

)

# --------------------------------------------------
# CAPACITY ASSUMPTIONS
# --------------------------------------------------

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

DEFAULT_CAPACITY = 700

with st.sidebar.expander("🏥 Capacity assumptions"):

    st.caption("Start with a base capacity, then adjust specific months if needed")

    # --- Base capacity ---
    base_capacity = st.number_input(
        "Base capacity (all months)",
        min_value=0,
        value=DEFAULT_CAPACITY,
        step=10,
        key="base_capacity"
    )

    # --- Track previous base ---
    if "prev_base_capacity" not in st.session_state:
        st.session_state.prev_base_capacity = base_capacity

    # --- Initialise monthly state ---
    for m in months:
        key = f"cap_{m}"
        if key not in st.session_state:
            st.session_state[key] = base_capacity

    # --- Only update months still equal to previous base ---
    if base_capacity != st.session_state.prev_base_capacity:
        prev_base = st.session_state.prev_base_capacity

        for m in months:
            key = f"cap_{m}"

            # Only update if user hasn't overridden
            if st.session_state[key] == prev_base:
                st.session_state[key] = base_capacity

        st.session_state.prev_base_capacity = base_capacity

    st.markdown("---")
    st.caption("Adjust individual months (highlighted if different from base)")

    # --- Monthly inputs with override highlighting ---
    monthly_capacity = {}

    for m in months:
        key = f"cap_{m}"
        value = st.session_state[key]

        is_override = value != base_capacity

        label = f"{m}"
        if is_override:
            label += " *"  # simple visual indicator

        monthly_capacity[m] = st.number_input(
            label,
            min_value=0,
            step=10,
            key=key
        )

    st.caption("* indicates values that differ from base capacity")


# --- Trend flexibility ---
st.sidebar.markdown("### Trend flexibility")

changepoint_scale = st.sidebar.slider(
    "How quickly should the trend adapt to change?",
    min_value=0.01,
    max_value=0.5,
    value=0.1,
    step=0.01,
    help=("""
        Trend flexibility controls how quickly the forecast reacts to real changes in demand
        - Lower values assume the service is broadly stable and smooth out short‑term fluctuations
        - Higher values allow the model to adapt quickly when demand has genuinely changed
        (e.g. service redesign, pathway changes, post‑COVID effects)
        
        Guidance:
        - 0.03–0.07 → Stable services, long‑term planning
        - 0.08–0.15 → Operational planning, gradual change
        - 0.15–0.30 → Known structural change or disruption
        
        There is no single 'correct' value – this should reflect service knowledge.
        """
    )
)

if changepoint_scale <= 0.07:
    st.sidebar.caption("🟢 Interpreting as: Stable service")
elif changepoint_scale <= 0.15:
    st.sidebar.caption("🟡 Interpreting as: Gradual change / operational planning")
else:
    st.sidebar.caption("🔴 Interpreting as: Structural change or disruption")

# --- Seasonality mode ---
st.sidebar.markdown("### Seasonality mode")

seasonality_mode = st.sidebar.selectbox(
    "How seasonal patterns scale with demand",
    ["additive", "multiplicative"],
    index=1,
    help=("""
        Controls how predictable seasonal patterns interact with overall demand.  
        
        Additive:
        - Seasonal effects add a fixed number of activities
        - Suitable when demand levels are stable  
        
        Multiplicative (recommended):
        - Seasonal effects scale with demand
        - Suitable when activities are growing or declining
        - Reflects how winter pressure often scales with activity levels
        """
    )
)

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (months)",
    min_value=3,
    max_value=36,
    value=12
)

# --------------------------------------------------
# MODEL FIT
# --------------------------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=changepoint_scale,
    seasonality_mode=seasonality_mode
)

model.fit(df)

future = model.make_future_dataframe(
    periods=forecast_horizon,
    freq="MS"
)

forecast = model.predict(future)

# --------------------------------------------------
# MAP CAPACITY TO FORECAST
# --------------------------------------------------

capacity_df = forecast[["ds"]].copy()
capacity_df["month"] = capacity_df["ds"].dt.month_name()

capacity_df["capacity"] = capacity_df["month"].map(monthly_capacity)

# --------------------------------------------------
# NHS COLOUR PALETTE
# --------------------------------------------------
NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_LIGHT_BLUE = "#41B6E6"
NHS_GREY = "#425563"

# --------------------------------------------------
# FORECAST PLOT
# --------------------------------------------------
st.subheader("📈 Forecast")

with st.expander("❓How to read the forecast"):

    st.caption(
        """
        - **Dots and solid line** show actual activity
        - The **shaded region** shows the forecast period
        - The central forecast line represents the **most likely** future path
        - The surrounding band reflects **plausible variation**, not error

        Actual activities are expected to fluctuate within this range rather than
        track the central line exactly

        **Capacity interpretation**

        The dashed red line represents estimated service capacity, which can vary by month.

        Where the forecast exceeds this line, there is a risk that demand may outstrip
        available capacity, potentially leading to:
        - Increased waiting times
        - Backlog growth
        - Operational pressure on services

        Capacity assumptions can be adjusted in the sidebar to explore different scenarios.
        """
    )

last_actual_date = df["ds"].iloc[-1]
forecast_end_date = forecast["ds"].iloc[-1]

fig = go.Figure()

# Forecast region shading
fig.add_vrect(
    x0=last_actual_date,
    x1=forecast_end_date,
    fillcolor="rgba(232, 237, 238, 0.6)",
    layer="below",
    line_width=0
)

# Actuals
fig.add_trace(go.Scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines+markers",
    name="Actual activities",
    line=dict(color=NHS_BLUE, width=2),
    marker=dict(size=6),
    hovertemplate="Actual: %{y:,.0f}<extra></extra>"
))

# Forecast (fitted + future)
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast",
    line=dict(color=NHS_DARK_BLUE, width=3),
    hovertemplate="Forecast: %{y:,.0f}<extra></extra>"
))

# Add dynamic capacity line

fig.add_trace(go.Scatter(
    x=capacity_df["ds"],
    y=capacity_df["capacity"],
    mode="lines",
    name="Capacity",
    line=dict(color="red", width=3, dash="dash"),
    hovertemplate="Capacity: %{y:,.0f}<extra></extra>"
))


# Confidence interval
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    line=dict(width=0),
    name="Upper bound",
    showlegend=False,
    hovertemplate="Upper bound: %{y:,.0f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    fill="tonexty",
    fillcolor="rgba(65, 182, 230, 0.3)",
    line=dict(width=0),
    name="Lower bound",
    hovertemplate="Lower bound: %{y:,.0f}<extra></extra>"
))

fig.add_vline(
    x=last_actual_date,
    line_width=2,
    line_dash="dash",
    line_color=NHS_GREY
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Number of activities",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5
    )
)

# --------------------------------------------------
# IDENTIFY CAPACITY BREACHES
# --------------------------------------------------

breach = forecast["yhat"] > capacity_df["capacity"]

# Overlay breach segments
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"].where(breach),
    mode="lines",
    line=dict(color="red", width=4),
    name="Above capacity",
    hovertemplate="Above capacity: %{y:,.0f}<extra></extra>"
))


st.plotly_chart(fig, width='stretch')




# --------------------------------------------------
# AUTO-GENERATED FORECAST NARRATIVE
# --------------------------------------------------

# Use recent forecast period for direction assessment
narrative_window = min(12, forecast_horizon)

recent_forecast = forecast.tail(narrative_window)

start_value = recent_forecast["yhat"].iloc[0]
end_value = recent_forecast["yhat"].iloc[-1]

pct_change = (end_value - start_value) / start_value

# Classify direction using conservative thresholds
if pct_change > 0.05:
    direction = "increase"
elif pct_change < -0.05:
    direction = "decrease"
else:
    direction = "stable"


# --------------------------------------------------
# COMPARISON TO LAST YEAR'S ACTUALS
# --------------------------------------------------

st.subheader("📊 Comparison to last year")

# Determine comparison window (up to 12 months)
comparison_months = min(12, forecast_horizon)

# Last year actuals
last_year_actuals = df.tail(comparison_months)

# Corresponding forecast period
forecast_period = forecast[forecast["ds"] > df["ds"].max()].head(comparison_months)

# Only proceed if both periods are available
if len(last_year_actuals) >= 3 and len(forecast_period) >= 3:

    last_year_mean = last_year_actuals["y"].mean()
    forecast_mean = forecast_period["yhat"].mean()

    yoy_change_pct = (forecast_mean - last_year_mean) / last_year_mean

    # Headline metrics    
    col_ly, col_fc= st.columns(2)

    with col_ly:
        st.metric(
            "Average last year",
            f"{last_year_mean:,.0f} activities / month"
        )

    with col_fc:
        st.metric(
            label="Forecast average",
            value=f"{forecast_mean:,.0f} activities / month",
            delta=f"{yoy_change_pct:+.1%}"
        )

else:
    st.info(
        "Not enough historical or forecast data is available to provide a "
        "robust comparison with last year."
    )

# --------------------------------------------------
# COMBINED FORECAST SUMMARY
# --------------------------------------------------

st.subheader("📝 Forecast interpretation")

st.caption(
    "This narrative is automatically generated from the forecast trend and should "
    "be interpreted alongside service knowledge and operational context."
)

trend_change = pct_change
avg_change = yoy_change_pct

# --- Scenario 1: Rising and above last year ---
if trend_change > 0 and avg_change > 0.05:

    summary_text = f"""
    **Demand is expected to increase over the forecast period.**

    - Activity volumes are projected to rise by approximately **{trend_change:.1%} overall**
    from the start to the end of the next {narrative_window} months.
    - On average, demand is expected to be around **{avg_change:.1%} higher than the most recent year**,
    indicating sustained upward pressure.

    This suggests both a **higher level of demand** and **continued growth**, with increasing
    capacity pressures over time.
    """

# --- Scenario 2: Falling but still above last year ---
elif trend_change < 0 and avg_change > 0.05:

    summary_text = f"""
    **Demand is expected to decrease over the forecast period, following a period of higher activity.**

    - Activity volumes are projected to change by approximately **{trend_change:.1%} overall**
    across the next {narrative_window} months.
    - However, average demand remains around **{avg_change:.1%} higher than the most recent year**,
    indicating that activity is still elevated compared to historical levels.

    This suggests that while pressures may ease over time, the service may still need to
    operate at **higher-than-usual capacity levels** in the near term.
    """

# --- Scenario 3: Rising but below last year ---
elif trend_change > 0 and avg_change < -0.05:

    summary_text = f"""
    **Demand is expected to increase over the forecast period, from a currently lower level.**

    - Activity volumes are projected to rise by approximately **{trend_change:.1%} overall**
    across the next {narrative_window} months.
    - Despite this upward trend, average demand is still expected to be around
    **{abs(avg_change):.1%} lower than the most recent year**.

    This suggests a **recovery in demand**, although overall activity may remain below
    previous levels.
    """

# --- Scenario 4: Falling and below last year ---
elif trend_change < 0 and avg_change < -0.05:

    summary_text = f"""
    **Demand is expected to decrease over the forecast period.**

    - Activity volumes are projected to change by approximately **{trend_change:.1%} overall**
    across the next {narrative_window} months.
    - On average, demand is expected to be around **{abs(avg_change):.1%} lower than the most recent year**,
    indicating a sustained reduction in activity.

    This suggests a **general easing of demand pressures**, although seasonal variation may remain.
    """

# --- Scenario 5: Stable / mixed ---
else:

    summary_text = f"""
    **Demand is expected to remain broadly stable over the forecast period.**

    - Activity volumes are projected to change by approximately **{trend_change:.1%} overall**
    across the next {narrative_window} months.
    - On average, demand is expected to be **within ±5% of the most recent year**.

    This suggests broadly stable activity levels, with variation driven mainly by
    seasonal patterns and short-term fluctuations.
    """

st.info(summary_text)


# --------------------------------------------------
# REQUIRED CAPACITY CALCULATION
# --------------------------------------------------

st.subheader("🧮 Required capacity to meet demand")

future_forecast = forecast[forecast["ds"] > df["ds"].max()]

# --- Option 1: Absolute requirement (no breaches) ---
max_required_capacity = future_forecast["yhat_upper"].max()

# --- Option 2: Typical planning level (e.g. 90th percentile) ---
p90_required_capacity = future_forecast["yhat"].quantile(0.90)

# --- Current baseline (average capacity) ---
avg_capacity = capacity_df["capacity"].mean()

# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current average capacity",
        f"{avg_capacity:,.0f}"
    )

with col2:
    st.metric(
        "Capacity to avoid all breaches",
        f"{max_required_capacity:,.0f}"
    )

with col3:
    st.metric(
        "Capacity for most months (90%)",
        f"{p90_required_capacity:,.0f}"
    )

st.info(f"""
    - To **avoid all capacity breaches**, the service would need capacity of approximately
    **{max_required_capacity:,.0f} activities per month**.
    - To meet demand in **most months (90%)**, capacity of approximately
    **{p90_required_capacity:,.0f} activities per month** would be sufficient.
""")

# --------------------------------------------------
# TREND AND SEASONALITY
# --------------------------------------------------

st.subheader("📊 Understanding demand patterns")

st.markdown("""
These charts show how demand behaves over time:

- **Trend** shows whether demand is increasing, stable, or decreasing over time  
- **Seasonality** shows which months tend to be consistently higher or lower than average

Together, they help distinguish between **sustained change in demand** and **predictable seasonal pressures**, supporting informed planning decisions.
""")

# --------------------------------------------------
# TREND PLOT
# --------------------------------------------------

trend_fig = go.Figure()

# Actuals
trend_fig.add_trace(go.Scatter(
    x=df["ds"],
    y=df["y"],
    mode="markers",
    name="Actual activities",
    marker=dict(size=6, color="#005EB8"),
    opacity=0.6
))

# Trend line
trend_fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["trend"],
    mode="lines",
    name="Underlying trend",
    line=dict(color="#003087", width=3)
))

# Forecast boundary
trend_fig.add_vline(
    x=df["ds"].max(),
    line_dash="dash",
    line_color="#425563",
    line_width=2
)

trend_fig.update_layout(
    title="Long‑term trend in demand",
    xaxis_title="Month",
    yaxis_title="activities per month",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25)
)

st.plotly_chart(trend_fig, width='stretch')

st.caption(
    "The trend reflects long‑term structural changes in activities, smoothing out "
    "short‑term variation. Use this to assess whether demand is increasing, stable, or decreasing."
)

# --------------------------------------------------
# MONTHLY SEASONALITY
# --------------------------------------------------

# Prepare monthly seasonality
seasonality_df = forecast[["ds", "yearly"]].copy()
seasonality_df["month"] = seasonality_df["ds"].dt.month_name()

month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

seasonality_df["month"] = pd.Categorical(
    seasonality_df["month"],
    categories=month_order,
    ordered=True
)

monthly_seasonality = (
    seasonality_df
    .groupby("month", observed=True)["yearly"]
    .mean()
    .reset_index()
)

# Convert to %
monthly_seasonality["pct_effect"] = monthly_seasonality["yearly"] * 100

# Identify peak and low months
peak_month = monthly_seasonality.loc[
    monthly_seasonality["pct_effect"].idxmax(), "month"
]

low_month = monthly_seasonality.loc[
    monthly_seasonality["pct_effect"].idxmin(), "month"
]

# Plot
seasonality_fig = go.Figure()

seasonality_fig.add_trace(go.Bar(
    x=monthly_seasonality["month"],
    y=monthly_seasonality["pct_effect"],
    marker_color="#41B6E6",
    name="Seasonal effect (%)"
))

seasonality_fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="#425563"
)

# Annotate peak month
seasonality_fig.add_annotation(
    x=peak_month,
    y=monthly_seasonality["pct_effect"].max(),
    text="Peak pressure",
    showarrow=True,
    arrowhead=1,
    yshift=10
)

# Annotate low month
seasonality_fig.add_annotation(
    x=low_month,
    y=monthly_seasonality["pct_effect"].min(),
    text="Lower demand",
    showarrow=True,
    arrowhead=1,
    yshift=-15
)

seasonality_fig.update_layout(
    title="Typical seasonal variation by month",
    xaxis_title="Month",
    yaxis_title="% above / below average month",
    showlegend=False
)

st.plotly_chart(seasonality_fig, width='stretch')

st.caption(
    "Values represent typical deviation from the average month. "
    "They reflect recurring seasonal patterns rather than one‑off events."
)

# --------------------------------------------------
# SEASONALITY NARRATIVE
# --------------------------------------------------

st.markdown(f"""
### 🍂Seasonal interpretation

Demand follows a consistent **within‑year pattern**:

- Demand tends to be **highest in {peak_month}**
- Demand tends to be **lowest in {low_month}**

This reflects predictable seasonal pressures rather than structural change.

Planning should ensure sufficient capacity during **peak months**, even if overall demand appears stable.
""")



# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------
st.subheader("⬇️ Download forecast output")

forecast_all = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].rename(
    columns={
        "ds": "Date",
        "yhat": "Forecast",
        "yhat_lower": "Lower",
        "yhat_upper": "Upper"
    }
)

actuals = df.rename(
    columns={
        "ds": "Date",
        "y": "Actual"
    }
)[["Date", "Actual"]]

forecast_output_df = forecast_all.merge(
    actuals,
    on="Date",
    how="left"
)

forecast_output_df["PeriodType"] = forecast_output_df["Actual"].apply(
    lambda x: "Actual" if pd.notna(x) else "Forecast"
)

forecast_output_df = forecast_output_df[
    ["Date", "Actual", "Forecast", "Lower", "Upper", "PeriodType"]
].sort_values("Date")

st.download_button(
    "Download forecast CSV",
    data=forecast_output_df.to_csv(index=False),
    file_name="forecast_output.csv",
    mime="text/csv"
)

# --------------------------------------------------
# ENSURE METRICS ARE AVAILABLE FOR REPORT
# --------------------------------------------------

# ---- Last year vs forecast ----
comparison_months = min(12, forecast_horizon)

last_year_actuals = df.tail(comparison_months)
forecast_period = forecast[forecast["ds"] > df["ds"].max()].head(comparison_months)

if len(last_year_actuals) >= 3 and len(forecast_period) >= 3:
    last_year_avg = last_year_actuals["y"].mean()
    forecast_avg = forecast_period["yhat"].mean()
    pct_change = (forecast_avg - last_year_avg) / last_year_avg
else:
    last_year_avg = 0
    forecast_avg = 0
    pct_change = 0

# ---- Capacity metrics ----
capacity_df = forecast[["ds"]].copy()
capacity_df["month"] = capacity_df["ds"].dt.month_name()
capacity_df["capacity"] = capacity_df["month"].map(monthly_capacity)

breach = forecast["yhat"] > capacity_df["capacity"]
breach_months = int(breach.sum())

excess_demand = (
    (forecast["yhat"] - capacity_df["capacity"])
    .clip(lower=0)
    .sum()
)

# ---- Required capacity ----
future_forecast = forecast[forecast["ds"] > df["ds"].max()]

max_required_capacity = future_forecast["yhat_upper"].max()
p90_required_capacity = future_forecast["yhat"].quantile(0.90)


# --------------------------------------------------
# HTML REPORT EXPORT (WITH INTERACTIVE CHARTS)
# --------------------------------------------------

st.subheader("📄 Export report")

def generate_html_report():

    clean_summary = textwrap.dedent(summary_text).strip()
    summary_html = markdown.markdown(clean_summary)

    forecast_chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    trend_chart_html = trend_fig.to_html(full_html=False, include_plotlyjs=False)
    seasonality_chart_html = seasonality_fig.to_html(full_html=False, include_plotlyjs=False)

    report_html = f"""
    <html>
    <head>
        <title>Demand Forecast Report</title>

        <!-- Load Plotly once -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                color: #222;
                line-height: 1.5;
            }}

            h1 {{
                color: #005EB8;
                border-bottom: 3px solid #005EB8;
                padding-bottom: 10px;
            }}

            h2 {{
                color: #003087;
                margin-top: 30px;
            }}

            .box {{
                background-color: #E8EDEE;
                padding: 15px;
                border-left: 6px solid #005EB8;
                margin-top: 10px;
            }}

            .metric {{
                margin: 4px 0;
                font-size: 15px;
            }}

            .chart {{
                margin-top: 20px;
            }}

            .footer {{
                margin-top: 40px;
                font-size: 12px;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 10px;
            }}
        </style>

    </head>

    <body>

        <h1>📈 Demand Forecast Report</h1>

        <!-- Executive Summary -->
        <h2>Executive Summary</h2>
        <div class="box">
            {summary_html}
        </div>

        <!-- Forecast Chart -->
        <h2>Forecast</h2>
        <div class="chart">
            {forecast_chart_html}
        </div>

        <!-- Comparison -->
        <h2>Comparison to Last Year</h2>
        <p class="metric"><b>Last year average:</b> {last_year_avg:,.0f}</p>
        <p class="metric"><b>Forecast average:</b> {forecast_avg:,.0f}</p>

        <!-- Capacity -->
        <h2>Capacity & Demand</h2>
        <p class="metric"><b>Average capacity:</b> {avg_capacity:,.0f}</p>
        <p class="metric"><b>Months above capacity:</b> {breach_months}</p>
        <p class="metric"><b>Total excess demand:</b> {excess_demand:,.0f}</p>

        <!-- Required Capacity -->
        <h2>Required Capacity</h2>
        <p class="metric"><b>To avoid all breaches:</b> {max_required_capacity:,.0f}</p>
        <p class="metric"><b>For 90% of months:</b> {p90_required_capacity:,.0f}</p>

        <!-- Trend Chart -->
        <h2>Trend Analysis</h2>
        <div class="chart">
            {trend_chart_html}
        </div>

        <!-- Seasonality Chart -->
        <h2>Seasonality</h2>
        <div class="chart">
            {seasonality_chart_html}
        </div>

        <p class="metric"><b>Peak month:</b> {peak_month}</p>
        <p class="metric"><b>Lowest month:</b> {low_month}</p>

        <div class="footer">
            This report is automatically generated and should be interpreted alongside 
            operational and clinical context.
        </div>

    </body>
    </html>
    """

    return report_html


if st.button("Generate report"):

    html_report = generate_html_report()

    st.download_button(
        label="⬇️ Download HTML report",
        data=html_report,
        file_name="demand_forecast_report.html",
        mime="text/html"
    )
