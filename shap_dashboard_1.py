"""
SHAP-Based Sales Analytics Dashboard
A professional, business-grade Streamlit application for non-technical users.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import time

warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Root Variables ── */
:root {
    --bg-main:    #F4F3EF;
    --bg-card:    #FFFFFF;
    --bg-panel:   #ECEAE4;
    --accent:     #1A2B4A;
    --accent2:    #2E5090;
    --accent3:    #C8943A;
    --text-main:  #111111;
    --text-sub:   #3A3A3A;
    --text-muted: #666666;
    --border:     #D4D0C8;
    --success:    #2A7A4B;
    --warn:       #B85C00;
    --radius:     10px;
    --radius-sm:  6px;
    --shadow-sm:  0 1px 4px rgba(0,0,0,0.07);
    --shadow:     0 3px 14px rgba(0,0,0,0.09);
    --shadow-md:  0 6px 24px rgba(0,0,0,0.11);
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg-main) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-main);
}

/* Smooth scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C4BFB6; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #A09890; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A2B4A 0%, #142038 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * {
    color: #E8E4D8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(200,148,58,0.15) !important;
    color: #E8E4D8 !important;
    border: 1px solid rgba(200,148,58,0.30) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.3rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em;
    font-size: 0.88rem !important;
    text-align: left !important;
    transition: background-color 0.18s, border-color 0.18s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(200,148,58,0.28) !important;
    border-color: rgba(200,148,58,0.55) !important;
}

/* ── Main Buttons ── */
.stButton > button {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.02em;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 6px rgba(26,43,74,0.25) !important;
    transition: background-color 0.18s, box-shadow 0.18s, transform 0.1s;
}
.stButton > button:hover {
    background-color: var(--accent2) !important;
    box-shadow: 0 4px 12px rgba(46,80,144,0.35) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0px); }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent) !important;
    letter-spacing: -0.01em;
}

/* ── Metric Cards — nuclear-strength override ── */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1.5px solid #D0CCC4 !important;
    border-top: 3px solid #1A2B4A !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 3px 14px rgba(0,0,0,0.09) !important;
}
/* Force ALL text inside metric cards to be dark and visible */
div[data-testid="metric-container"],
div[data-testid="metric-container"] *,
div[data-testid="metric-container"] p,
div[data-testid="metric-container"] span,
div[data-testid="metric-container"] div,
div[data-testid="metric-container"] label {
    color: #111111 !important;
    background-color: transparent !important;
    opacity: 1 !important;
}
/* Label — smaller caps */
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] *,
div[data-testid="metric-container"] label {
    color: #444444 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
/* Value — large bold number */
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #0D1F38 !important;
    font-weight: 800 !important;
    font-size: 1.7rem !important;
}
/* Delta */
div[data-testid="metric-container"] [data-testid="stMetricDelta"],
div[data-testid="metric-container"] [data-testid="stMetricDelta"] * {
    font-weight: 700 !important;
    color: #2E5090 !important;
}

/* ── Dataframes ── */
.stDataFrame {
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 0.9rem !important;
    transition: border-color 0.18s, box-shadow 0.18s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 0 3px rgba(46,80,144,0.12) !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Info / success / warning boxes ── */
.insight-box {
    background: linear-gradient(135deg, #EEF2F8 0%, #E8EDF5 100%);
    border-left: 4px solid var(--accent2);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1.1rem;
    margin-top: 0.5rem;
    color: var(--text-sub);
    font-size: 0.87rem;
    line-height: 1.55;
    box-shadow: var(--shadow-sm);
}
.success-box {
    background: linear-gradient(135deg, #EAF5EE 0%, #E2F0E8 100%);
    border-left: 4px solid var(--success);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    color: #1A4A30;
    font-size: 0.87rem;
    box-shadow: var(--shadow-sm);
}
.warn-box {
    background: linear-gradient(135deg, #FDF3E7 0%, #F8ECD8 100%);
    border-left: 4px solid var(--accent3);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    color: #6B3C00;
    font-size: 0.87rem;
    box-shadow: var(--shadow-sm);
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--accent);
    border-bottom: 2px solid var(--accent3);
    padding-bottom: 0.4rem;
    margin-bottom: 1.3rem;
    letter-spacing: -0.01em;
}
.step-badge {
    display: inline-block;
    background-color: var(--accent);
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent2) !important;
}
/* hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: styled chart caption ─────────────────────────────────────────────
def chart_caption(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


# ── Helper: KPI metric card (guaranteed visible, bypasses Streamlit st.metric) ─
def kpi_metric(col, label: str, value, delta: str = None):
    """Render a KPI card with solid black text — immune to Streamlit theme issues."""
    delta_html = ""
    if delta is not None:
        delta_color = "#2A7A4B" if str(delta).startswith("+") else ("#B03A2E" if str(delta).startswith("-") else "#2E5090")
        delta_html = f"""
        <div style='font-size:0.82rem; font-weight:700; color:{delta_color};
                    margin-top:0.25rem;'>{delta}</div>"""
    col.markdown(f"""
    <div style='background:#FFFFFF; border:1.5px solid #D0CCC4;
                border-top:3px solid #1A2B4A; border-radius:10px;
                padding:1.1rem 1.3rem; box-shadow:0 1px 4px rgba(0,0,0,0.07);'>
        <div style='font-size:0.70rem; font-weight:700; letter-spacing:0.08em;
                    text-transform:uppercase; color:#555555;
                    margin-bottom:0.35rem;'>{label}</div>
        <div style='font-size:1.7rem; font-weight:800; color:#0D1F38;
                    line-height:1.15;'>{value}</div>{delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Helper: cached file reader — reads ONCE per unique file, then instant ────
@st.cache_data(show_spinner=False)
def _read_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Read CSV or Excel from raw bytes. Result cached by content hash."""
    import io
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".csv"):
        return pd.read_csv(buf, low_memory=False)
    else:
        # calamine (Rust) is 3-5× faster than openpyxl on large xlsx
        try:
            return pd.read_excel(buf, engine="calamine")
        except Exception:
            buf.seek(0)
            return pd.read_excel(buf, engine="openpyxl")
PALETTE = ["#1A2B4A", "#2E5090", "#C8943A", "#5B8AC9", "#8AB4D4", "#E8C07A"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FAFAF8")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#111111", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color="#3A3A3A", labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, color="#3A3A3A", labelpad=6)
    ax.tick_params(colors="#3A3A3A", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D8D4CC")
    ax.grid(axis="y", color="#E8E4DC", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)


def styled_fig(figsize=(10, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")
    return fig, ax


# ── Preprocessing ─────────────────────────────────────────────────────────────
EXPECTED_COLS = [
    "DistributorCode", "SRCode", "Date", "TimeIn", "TimeOut", "CallDuration",
    "ChannelType", "Region", "Area", "OutletID", "OutletName", "Classification",
    "RouteID", "VisitStatus", "VisitLatitude", "VisitLongitude",
    "OutletLatitude", "OutletLongitude", "ProductID", "ProductCategory",
    "SalesQty", "SalesValue", "DeliveryQty", "DeliveryDelay"
]

def run_preprocessing(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # ── 1. Standardise column names ──
    df.columns = [c.strip() for c in df.columns]

    # ── 2. Add missing expected columns ──
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # ── 3. Parse Date ──
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Month"]     = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["WeekNum"]   = df["Date"].dt.isocalendar().week.astype(int)

    # ── 4. Parse time fields ──
    def parse_hms(s):
        try:
            parts = str(s).split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except Exception:
            return np.nan

    if "TimeIn" in df.columns:
        df["TimeInSec"]  = df["TimeIn"].apply(parse_hms)
    if "TimeOut" in df.columns:
        df["TimeOutSec"] = df["TimeOut"].apply(parse_hms)
    if "CallDuration" in df.columns:
        df["CallDuration"] = pd.to_numeric(df["CallDuration"], errors="coerce")

    # ── 5. Numeric coercion ──
    num_cols = ["SalesQty", "SalesValue", "DeliveryQty", "DeliveryDelay",
                "VisitLatitude", "VisitLongitude", "OutletLatitude", "OutletLongitude"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 6. Fill numeric nulls with median ──
    num_fill = ["SalesQty", "SalesValue", "DeliveryQty", "DeliveryDelay",
                "CallDuration", "TimeInSec", "TimeOutSec"]
    for c in num_fill:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med if not np.isnan(med) else 0)

    # ── 7. Geo-distance feature (haversine km) ──
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    geo_cols = ["VisitLatitude", "VisitLongitude", "OutletLatitude", "OutletLongitude"]
    if all(c in df.columns for c in geo_cols):
        mask = df[geo_cols].notna().all(axis=1)
        df["GeoDeviation_km"] = np.nan
        if mask.any():
            df.loc[mask, "GeoDeviation_km"] = haversine(
                df.loc[mask, "VisitLatitude"],
                df.loc[mask, "VisitLongitude"],
                df.loc[mask, "OutletLatitude"],
                df.loc[mask, "OutletLongitude"],
            )
        df["GeoDeviation_km"] = df["GeoDeviation_km"].fillna(0)

    # ── 8. VisitSuccess binary flag ──
    if "VisitStatus" in df.columns:
        df["VisitSuccess"] = df["VisitStatus"].astype(str).str.upper().str.startswith("E").astype(int)

    # ── 9. Encode categorical columns ──
    cat_encode = ["ChannelType", "Region", "Area", "Classification", "ProductCategory"]
    for c in cat_encode:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")
            df[c + "_Code"] = pd.Categorical(df[c]).codes

    # ── 10. Drop raw columns we won't use as features ──
    drop_cols = ["Date", "TimeIn", "TimeOut", "OutletID", "OutletName",
                 "VisitStatus", "ProductID", "RouteID", "SRCode",
                 "ChannelType", "Region", "Area", "Classification", "ProductCategory"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 11. Drop non-numeric remnants ──
    df = df.select_dtypes(include=[np.number])
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(0, inplace=True)

    return df


# ── Model Training & SHAP ──────────────────────────────────────────────────
def train_model(df: pd.DataFrame, target: str, algo: str = "Gradient Boosting"):
    from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                                  RandomForestRegressor, RandomForestClassifier)
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (mean_absolute_error, r2_score,
                                 accuracy_score, roc_auc_score, f1_score)

    X = df.drop(columns=[target])
    y = df[target]

    is_clf = (y.nunique() <= 5 and y.dtype in [int, "int64", "int32"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    MODEL_MAP_CLF = {
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    }
    MODEL_MAP_REG = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
    }

    model_map = MODEL_MAP_CLF if is_clf else MODEL_MAP_REG
    model = model_map.get(algo, list(model_map.values())[0])
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    if is_clf:
        metrics = {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        }
        try:
            metrics["AUC-ROC"] = round(
                roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1]), 4
            )
        except Exception:
            pass
    else:
        metrics = {
            "R² Score": round(r2_score(y_test, y_pred), 4),
            "MAE":      round(mean_absolute_error(y_test, y_pred), 4),
        }

    return model, scaler, X_train, X_test, y_test, y_pred, metrics, X.columns.tolist(), is_clf


def _compare_all_models(df: pd.DataFrame, target: str):
    """Train all four algorithms and return a comparison DataFrame."""
    from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                                  RandomForestRegressor, RandomForestClassifier)
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

    X = df.drop(columns=[target])
    y = df[target]
    is_clf = (y.nunique() <= 5 and y.dtype in [int, "int64", "int32"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if is_clf:
        candidates = {
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        }
    else:
        candidates = {
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Ridge Regression":  Ridge(alpha=1.0),
            "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
        }

    rows = []
    for name, m in candidates.items():
        m.fit(X_train_s, y_train)
        yp = m.predict(X_test_s)
        if is_clf:
            row = {"Algorithm": name,
                   "Accuracy":  round(accuracy_score(y_test, yp), 4),
                   "F1 Score":  round(f1_score(y_test, yp, average="weighted"), 4)}
        else:
            row = {"Algorithm": name,
                   "R² Score":  round(r2_score(y_test, yp), 4),
                   "MAE":       round(mean_absolute_error(y_test, yp), 4)}
        rows.append(row)

    return pd.DataFrame(rows), is_clf


def compute_shap(model, X_train, X_test, feature_names):
    try:
        import shap
        explainer   = shap.Explainer(model, X_train)
        shap_values = explainer(X_test, check_additivity=False)
        return shap_values, explainer
    except ImportError:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:"DM Serif Display",serif; font-size:1.3rem;
                    color:#E8E4D8; letter-spacing:0.02em;'>
            Sales Analytics<br>Intelligence
        </div>
        <div style='font-size:0.72rem; color:#A8A090; margin-top:0.3rem;
                    letter-spacing:0.08em; text-transform:uppercase;'>
            SHAP-Driven Insights
        </div>
        <hr style='border-color:#2E4070; margin:1rem 0;'>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "Data Upload",
        "Preprocessing",
        "Exploratory Analysis",
        "Model Training",
        "SHAP Analysis",
        "Performance Evaluation",
        "Store Performance",
        "AI Assistant",
    ]

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    for opt in nav_options:
        active_style = "border-left: 3px solid #C8943A; padding-left:8px;" if st.session_state.page == opt else ""
        if st.button(opt, key=f"nav_{opt}"):
            st.session_state.page = opt

    st.markdown("""
    <hr style='border-color:#2E4070; margin:1.5rem 0 0.8rem;'>
    <div style='font-size:0.72rem; color:#706858; text-align:center;
                line-height:1.6; letter-spacing:0.04em;'>
        Gradient Boosting + SHAP<br>
        Professional Analytics Suite
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div style='padding: 2rem 0 1rem;'>
        <div style='font-family:"DM Serif Display",serif; font-size:2.2rem;
                    color:#1A2B4A; line-height:1.2;'>
            Sales Analytics Intelligence
        </div>
        <div style='font-size:1rem; color:#555; margin-top:0.4rem;'>
            A SHAP-powered decision support platform for distributor sales performance
        </div>
        <hr style='border-color:#D8D4CC; margin: 1.5rem 0;'>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #D8D4CC; border-radius:8px;
                    padding:1.4rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                        color:#888; text-transform:uppercase;'>Step 1 — 2</div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                        color:#1A2B4A; margin:0.5rem 0 0.3rem;'>Data Ingestion</div>
            <div style='font-size:0.85rem; color:#555; line-height:1.5;'>
                Upload your distributor sales dataset. The system validates structure
                and provides an immediate preview with quality diagnostics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #D8D4CC; border-radius:8px;
                    padding:1.4rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                        color:#888; text-transform:uppercase;'>Step 3 — 5</div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                        color:#1A2B4A; margin:0.5rem 0 0.3rem;'>Analytics Pipeline</div>
            <div style='font-size:0.85rem; color:#555; line-height:1.5;'>
                One-click preprocessing, exploratory charts, and automated model
                training — no configuration required.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #D8D4CC; border-radius:8px;
                    padding:1.4rem; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                        color:#888; text-transform:uppercase;'>Step 6 — 7</div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.1rem;
                        color:#1A2B4A; margin:0.5rem 0 0.3rem;'>SHAP Insights</div>
            <div style='font-size:0.85rem; color:#555; line-height:1.5;'>
                Interpretable AI explanations reveal which factors drive sales
                performance, with visual evidence at every level.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Workflow at a Glance</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Upload Dataset",        "Load a CSV or Excel file containing distributor sales records."),
        ("02", "Data Preview",           "Inspect raw data, column coverage, and data quality metrics."),
        ("03", "Standard Preprocessing","One action handles nulls, encoding, feature engineering, and scaling."),
        ("04", "Exploratory Analysis",   "Visual summaries of sales trends, channel mix, and regional patterns."),
        ("05", "Model Training",         "Gradient Boosting model trained automatically with cross-validation."),
        ("06", "SHAP Analysis",          "Feature importance, dependency plots, and individual-record explanations."),
        ("07", "Performance Evaluation", "Regression or classification metrics with residual diagnostics."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:1rem; margin-bottom:0.75rem;
                    background:#FFFFFF; border:1px solid #E8E4DC; border-radius:8px;
                    padding:0.9rem 1.1rem;'>
            <div style='font-family:"DM Serif Display",serif; font-size:1.4rem;
                        color:#C8943A; min-width:2rem; font-weight:400;'>{num}</div>
            <div>
                <div style='font-weight:700; font-size:0.93rem; color:#1A2B4A;'>{title}</div>
                <div style='font-size:0.83rem; color:#555; margin-top:0.2rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="insight-box">Begin by navigating to <strong>Data Upload</strong> in the sidebar to load your dataset.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Data Upload":
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#555; font-size:0.9rem;">Upload a CSV or Excel file containing your distributor sales records. The system accepts any dataset that includes the standard column schema.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select File",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx / .xls)"
    )

    if uploaded:
        # Read bytes once; cache_data returns instantly on repeated renders
        file_bytes = uploaded.read()
        with st.spinner("Loading file…"):
            try:
                df_raw = _read_uploaded_file(file_bytes, uploaded.name)
                st.session_state["df_raw"] = df_raw
                st.session_state.pop("df_processed", None)
                st.session_state.pop("model_result", None)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

        st.markdown('<div class="success-box">File loaded successfully.</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        missing_pct = round(df_raw.isnull().mean().mean() * 100, 1)
        # Duplicate check: sample first 50 k rows on large files to keep it fast
        _dup_sample = df_raw if len(df_raw) <= 50_000 else df_raw.sample(50_000, random_state=0)
        dup_count   = _dup_sample.duplicated().sum()
        dup_label   = f"{dup_count:,}" if len(df_raw) <= 50_000 else f"~{dup_count:,}*"
        kpi_metric(col1, "Total Records", f"{len(df_raw):,}")
        kpi_metric(col2, "Columns", len(df_raw.columns))
        kpi_metric(col3, "Missing Values", f"{missing_pct}%")
        kpi_metric(col4, "Duplicate Rows", dup_label)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Data Preview (first 10 rows)**")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # Column coverage report
        st.markdown("<br>**Column Coverage Against Expected Schema**")
        present = [c for c in EXPECTED_COLS if c in df_raw.columns]
        absent  = [c for c in EXPECTED_COLS if c not in df_raw.columns]
        coverage_df = pd.DataFrame({
            "Column":   EXPECTED_COLS,
            "Present":  ["Yes" if c in df_raw.columns else "No" for c in EXPECTED_COLS],
            "Non-Null Count": [
                int(df_raw[c].notna().sum()) if c in df_raw.columns else 0
                for c in EXPECTED_COLS
            ],
        })
        st.dataframe(coverage_df, use_container_width=True, height=320)

        if absent:
            st.markdown(f'<div class="warn-box"><strong>Note:</strong> {len(absent)} expected column(s) not found in this file — they will be added as empty columns during preprocessing: {", ".join(absent)}</div>', unsafe_allow_html=True)

        chart_caption("This table confirms which columns from the expected schema are present in your file. Missing columns are automatically added with blank values during preprocessing so the pipeline always runs consistently.")

        st.markdown("<br>")
        st.info("Proceed to **Preprocessing** in the sidebar to prepare this dataset for analysis.")

    else:
        st.markdown("""
        <div style='background:#F7F6F2; border:2px dashed #C8B898; border-radius:8px;
                    padding:2.5rem; text-align:center;'>
            <div style='font-size:0.9rem; color:#777;'>
                No file uploaded yet. Use the selector above to load your dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Preprocessing":
    st.markdown('<div class="section-header">Standard Preprocessing</div>', unsafe_allow_html=True)

    if "df_raw" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()

    df_raw = st.session_state["df_raw"]

    st.markdown("""
    <p style='font-size:0.9rem; color:#444; line-height:1.6;'>
    All preprocessing steps are executed automatically with a single action.
    This includes null handling, categorical encoding, date and time feature extraction,
    geospatial deviation calculation, and numeric scaling preparation.
    </p>
    """, unsafe_allow_html=True)

    steps_info = [
        ("Column Standardisation",     "Ensures column names are clean and all expected fields are present."),
        ("Date & Time Extraction",     "Parses dates into Month, Week, and Day-of-Week; converts time to seconds."),
        ("Null Imputation",            "Numeric nulls filled with column medians; categoricals filled with 'Unknown'."),
        ("Geo-Deviation Feature",      "Calculates distance (km) between GPS visit location and actual outlet."),
        ("Visit Success Flag",         "Creates a binary target based on whether a visit was a successful call."),
        ("Categorical Encoding",       "Converts text categories (Region, ChannelType, etc.) to numeric codes."),
        ("Feature Finalisation",       "Drops non-numeric and identifier columns; retains only model-ready features."),
    ]

    for i, (step, desc) in enumerate(steps_info, 1):
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; align-items:flex-start;
                    margin-bottom:0.5rem; padding:0.65rem 1rem;
                    background:#FFFFFF; border:1px solid #E8E4DC; border-radius:6px;'>
            <div style='background:#1A2B4A; color:#FFF; font-size:0.72rem; font-weight:700;
                        border-radius:3px; padding:0.15rem 0.45rem; min-width:1.8rem;
                        text-align:center; margin-top:0.1rem;'>{i}</div>
            <div>
                <div style='font-weight:700; font-size:0.88rem; color:#1A2B4A;'>{step}</div>
                <div style='font-size:0.82rem; color:#666;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>")
    if st.button("Run Standard Preprocessing", use_container_width=True):
        with st.spinner("Preprocessing in progress..."):
            time.sleep(0.8)
            df_proc = run_preprocessing(df_raw)
            st.session_state["df_processed"] = df_proc

        st.markdown('<div class="success-box">Preprocessing complete. Dataset is ready for analysis.</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        kpi_metric(col1, "Features (after prep)", len(df_proc.columns))
        kpi_metric(col2, "Records Retained",      f"{len(df_proc):,}")
        kpi_metric(col3, "Remaining Nulls",        df_proc.isnull().sum().sum())

        st.markdown("<br>**Processed Dataset Preview**")
        st.dataframe(df_proc.head(10), use_container_width=True)

        st.markdown("<br>**Feature Summary Statistics**")
        st.dataframe(df_proc.describe().T.round(3), use_container_width=True)

        chart_caption("The processed dataset contains only numeric, model-ready features. All categorical variables have been converted to numeric codes, and new features (such as geo-deviation and time-based fields) have been added to improve predictive power.")

    elif "df_processed" in st.session_state:
        df_proc = st.session_state["df_processed"]
        st.markdown('<div class="success-box">Preprocessing was already completed. Showing existing results.</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        kpi_metric(col1, "Features", len(df_proc.columns))
        kpi_metric(col2, "Records",  f"{len(df_proc):,}")
        kpi_metric(col3, "Nulls",    df_proc.isnull().sum().sum())
        st.dataframe(df_proc.head(10), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: EXPLORATORY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    st.markdown('<div class="section-header">Exploratory Analysis</div>', unsafe_allow_html=True)

    if "df_raw" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()

    df_raw = st.session_state["df_raw"]
    df_proc = st.session_state.get("df_processed", None)

    # ── Chart 1: Visit Status Distribution ──────────────────────────────────
    if "VisitStatus" in df_raw.columns:
        st.markdown("**Visit Status Distribution**")
        vc = df_raw["VisitStatus"].value_counts().head(8)
        fig, ax = styled_fig((10, 4))
        bars = ax.barh(vc.index[::-1], vc.values[::-1], color=PALETTE[0], edgecolor="none", height=0.55)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + vc.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", ha="left", fontsize=8.5, color="#333")
        style_ax(ax, "Visit Status Distribution", "Number of Visits", "Status Category")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("This chart shows how visits are distributed across different outcome categories. A higher proportion of successful (E-type) visits generally indicates better field execution and outlet engagement quality.")

    # ── Chart 2: Sales Qty by Region ─────────────────────────────────────────
    if "Region" in df_raw.columns and "SalesQty" in df_raw.columns:
        st.markdown("<br>**Sales Quantity by Region**")
        grp = df_raw.groupby("Region")["SalesQty"].sum().sort_values(ascending=False).head(10)
        fig, ax = styled_fig((10, 4))
        ax.bar(grp.index, grp.values, color=PALETTE[1], edgecolor="none", width=0.55)
        style_ax(ax, "Total Sales Quantity by Region", "Region", "Sales Quantity (Units)")
        plt.xticks(rotation=30, ha="right", fontsize=8.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Regional comparison of total sales volume. Regions with significantly higher quantities may indicate stronger distribution networks, better SR performance, or higher market demand in those areas.")
        st.markdown("""
        <div style='background:#F7F6F2; border:1px solid #D8D4CC; border-radius:6px;
                    padding:0.85rem 1.1rem; margin-top:0.6rem;'>
            <div style='font-weight:700; font-size:0.88rem; color:#1A2B4A; margin-bottom:0.35rem;'>
                📍 How Regions Are Defined
            </div>
            <div style='font-size:0.84rem; color:#333333; line-height:1.65;'>
                Regions in this dataset represent <strong>pre-defined geographic sales territories</strong>
                assigned by the distributor or company. Each region groups a set of outlets
                (retail stores, supermarkets, kiosks, etc.) that fall within a common geographic boundary —
                such as a city cluster, state zone, or district block.<br><br>
                The region label for each record is sourced directly from the <code>Region</code> column
                in your uploaded dataset. Boundaries reflect the territories defined by your field
                operations or distribution management team for route planning and SR (Sales Representative)
                assignment. Regions are <strong>not derived or recalculated</strong> by this system —
                they are taken as-is from the raw data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chart 3: Channel Type Mix ─────────────────────────────────────────────
    if "ChannelType" in df_raw.columns:
        st.markdown("<br>**Channel Type Mix**")
        ct = df_raw["ChannelType"].value_counts()
        fig, ax = styled_fig((7, 4))
        wedges, texts, autotexts = ax.pie(
            ct.values, labels=ct.index, autopct="%1.1f%%",
            colors=PALETTE[:len(ct)], startangle=140,
            pctdistance=0.82, wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(8.5)
            at.set_color("#111")
            at.set_fontweight("bold")
        ax.set_title("Channel Type Distribution", fontsize=13, fontweight="bold", color="#111", pad=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("The breakdown of visits and sales across channel types (e.g., Large Traditional, Modern Trade). Understanding channel mix helps prioritise where SR effort should be directed for maximum sales impact.")

    # ── Chart 4: Call Duration Distribution ───────────────────────────────────
    if "CallDuration" in df_raw.columns:
        st.markdown("<br>**Call Duration Distribution**")
        cd = pd.to_numeric(df_raw["CallDuration"], errors="coerce").dropna()
        cd_clipped = cd[(cd >= -100) & (cd <= 100)]
        fig, ax = styled_fig((10, 4))
        ax.hist(cd_clipped, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.6, alpha=0.9,
                range=(-100, 100))
        style_ax(ax, "Distribution of Call Duration", "Duration (minutes)", "Number of Visits")
        ax.set_xlim(-100, 100)
        median_val = cd_clipped.median() if len(cd_clipped) > 0 else cd.median()
        ax.axvline(median_val, color=PALETTE[2], linewidth=1.8, linestyle="--",
                   label=f"Median: {median_val:.1f} min")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("The histogram shows how long sales representatives spend at each outlet. The X-axis spans −100 to +100 minutes to keep the view focused on the core distribution. The dashed line marks the median visit duration. Visits that are too short may indicate insufficient engagement, while very long visits may indicate inefficiency.")

    # ── Chart 5: Sales Value by Classification ────────────────────────────────
    if "Classification" in df_raw.columns and "SalesQty" in df_raw.columns:
        st.markdown("<br>**Sales Quantity by Outlet Classification**")
        cls_grp = df_raw.groupby("Classification")["SalesQty"].agg(["mean", "sum"]).reset_index()
        cls_grp.columns = ["Classification", "Avg_SalesQty", "Total_SalesQty"]
        cls_grp.sort_values("Total_SalesQty", ascending=False, inplace=True)
        fig, ax = styled_fig((10, 4))
        bars = ax.bar(cls_grp["Classification"], cls_grp["Total_SalesQty"],
                      color=PALETTE[2], edgecolor="none", width=0.5)
        style_ax(ax, "Total Sales Quantity by Outlet Classification", "Classification Tier", "Total Sales Quantity (Units)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Outlet classifications (e.g., Gold, Silver, Bronze) segment accounts by strategic importance. This chart reveals which tiers generate the most volume, helping align SR visit priorities with revenue potential.")

    # ── Chart 6: Monthly trend ─────────────────────────────────────────────────
    if "Date" in df_raw.columns and "SalesQty" in df_raw.columns:
        df_raw_date = df_raw.copy()
        df_raw_date["Date"] = pd.to_datetime(df_raw_date["Date"], dayfirst=True, errors="coerce")
        df_raw_date["SalesQty"] = pd.to_numeric(df_raw_date["SalesQty"], errors="coerce")
        monthly = df_raw_date.groupby(df_raw_date["Date"].dt.to_period("M"))["SalesQty"].sum().dropna()
        if len(monthly) > 1:
            st.markdown("<br>**Monthly Sales Quantity Trend**")
            fig, ax = styled_fig((11, 4))
            ax.plot(range(len(monthly)), monthly.values, color=PALETTE[0],
                    linewidth=2.2, marker="o", markersize=5, markerfacecolor=PALETTE[2])
            ax.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color=PALETTE[0])
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels([str(p) for p in monthly.index], rotation=30, ha="right", fontsize=8)
            style_ax(ax, "Monthly Sales Quantity Trend", "Month", "Total Sales Quantity (Units)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            chart_caption("This trend line tracks total sales volume across months. Rising trends may reflect successful promotions or seasonal demand; drops may signal coverage gaps, stock issues, or reduced SR activity.")

    # ── Correlation Heatmap (processed) ───────────────────────────────────────
    if df_proc is not None and len(df_proc.columns) >= 3:
        st.markdown("<br>**Feature Correlation Matrix**")
        corr = df_proc.corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)),
                               facecolor="#FFFFFF")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(corr) <= 12, fmt=".2f", ax=ax,
                    cmap="Blues", linewidths=0.4, linecolor="#E8E4DC",
                    annot_kws={"size": 7.5, "color": "#111"},
                    cbar_kws={"shrink": 0.7})
        ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold",
                     color="#111", pad=12)
        ax.tick_params(labelsize=8, colors="#333")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("The correlation matrix shows relationships between all processed features. Strong positive correlations (dark blue) indicate features that move together. Features highly correlated with the target variable tend to be the most predictive.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)

    if "df_processed" not in st.session_state:
        st.warning("Complete preprocessing before training the model.")
        st.stop()

    df_proc = st.session_state["df_processed"]
    cols = df_proc.columns.tolist()

    st.markdown("""
    <p style='font-size:0.9rem; color:#444; line-height:1.6;'>
    Select the column you want to predict and choose a machine learning algorithm.
    The system detects whether to use a regression or classification approach based on the target variable.
    </p>
    """, unsafe_allow_html=True)

    # ── Side-by-side dropdowns ────────────────────────────────────────────────
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        target = st.selectbox("Select Target Variable", options=cols,
                              index=cols.index("SalesQty") if "SalesQty" in cols else 0)
    with sel_col2:
        is_clf_preview = (df_proc[target].nunique() <= 5 and
                          df_proc[target].dtype in [int, "int64", "int32"])
        algo_options = (["Gradient Boosting", "Random Forest", "Logistic Regression", "Decision Tree"]
                        if is_clf_preview else
                        ["Gradient Boosting", "Random Forest", "Ridge Regression", "Decision Tree"])
        algo = st.selectbox("Select Algorithm", options=algo_options)

    ALGO_DESCRIPTIONS = {
        "Gradient Boosting":   "Builds trees sequentially, each correcting the last. Generally the most accurate.",
        "Random Forest":       "Many independent trees averaged together. Fast, robust, handles noisy data well.",
        "Ridge Regression":    "Linear model with regularisation. Very fast and interpretable — ideal as a baseline.",
        "Logistic Regression": "Linear classifier. Fast and interpretable when classes are well-separated.",
        "Decision Tree":       "A single decision tree. Highly interpretable but may overfit without pruning.",
    }

    st.markdown(f"""
    <div style='background:#EEF2F8; border:1px solid #C8D8EC; border-radius:6px;
                padding:0.75rem 1rem; font-size:0.86rem; color:#2E4070; margin:0.5rem 0 1rem;'>
        <strong>Target:</strong> {target} &nbsp;|&nbsp;
        <strong>Features:</strong> {len(cols) - 1} columns &nbsp;|&nbsp;
        <strong>Task:</strong> {"Classification" if is_clf_preview else "Regression"} &nbsp;|&nbsp;
        <strong>Algorithm:</strong> {algo}<br>
        <span style='color:#555; font-size:0.82rem;'>{ALGO_DESCRIPTIONS.get(algo, "")}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training in progress. This may take a moment..."):
            result = train_model(df_proc, target, algo)
            st.session_state["model_result"] = result
            st.session_state["target"] = target
            st.session_state["trained_algo"] = algo

        model, scaler, X_train, X_test, y_test, y_pred, metrics, feature_names, is_clf = result

        st.markdown('<div class="success-box">Model trained successfully.</div>', unsafe_allow_html=True)

        task_label = "Classification" if is_clf else "Regression"
        st.markdown(f"""
        <div style='background:#F7F6F2; border:1px solid #D8D4CC; border-radius:6px;
                    padding:0.75rem 1rem; font-size:0.86rem; color:#333; margin-bottom:1rem;'>
            <strong>Task Type Detected:</strong> {task_label} &mdash;
            {"Target has 5 or fewer unique values (treated as class labels)." if is_clf
             else "Target is continuous numeric (treated as a quantity to predict)."}
        </div>
        """, unsafe_allow_html=True)

        cols_m = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            kpi_metric(cols_m[i], k, v)
        st.markdown("<br>**Feature Importance (Top 15)**")
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
            fig, ax = styled_fig((10, 5))
            ax.barh(fi.index[::-1], fi.values[::-1], color=PALETTE[1], edgecolor="none", height=0.6)
            style_ax(ax, f"Feature Importance — Top 15 Predictors ({algo})", "Importance Score", "Feature")
            ax.set_xlim(0, fi.max() * 1.15)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            chart_caption("Feature importance scores indicate which variables the model relies on most when making predictions. Higher scores mean the model splits the data on that feature more frequently and with greater effect. These scores form the foundation for the SHAP analysis on the next page.")
        else:
            st.info("This algorithm does not expose feature importances directly. Use SHAP Analysis for interpretability.")

    elif "model_result" in st.session_state:
        model, scaler, X_train, X_test, y_test, y_pred, metrics, feature_names, is_clf = st.session_state["model_result"]
        trained_algo = st.session_state.get("trained_algo", "Unknown")
        st.markdown(f'<div class="success-box">Model already trained ({trained_algo}). Showing existing results.</div>', unsafe_allow_html=True)
        cols_m = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            kpi_metric(cols_m[i], k, v)
    st.markdown("<hr style='border-color:#D8D4CC; margin: 2rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="font-size:1.25rem;">Algorithm Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.88rem; color:#444; line-height:1.6;'>
    Benchmark all available algorithms on your dataset simultaneously.
    The best-performing model is highlighted — use this to confirm your algorithm choice before training.
    </p>
    """, unsafe_allow_html=True)

    if st.button("Run Full Algorithm Comparison", use_container_width=True):
        with st.spinner("Benchmarking all algorithms… this may take 30–60 seconds."):
            comp_df, comp_is_clf = _compare_all_models(df_proc, target)
            st.session_state["comparison_df"] = comp_df
            st.session_state["comp_is_clf"]   = comp_is_clf

    if "comparison_df" in st.session_state:
        comp_df        = st.session_state["comparison_df"]
        comp_is_clf    = st.session_state["comp_is_clf"]
        primary_metric = "Accuracy" if comp_is_clf else "R² Score"
        best_algo      = comp_df.loc[comp_df[primary_metric].idxmax(), "Algorithm"]
        best_val       = comp_df.loc[comp_df["Algorithm"] == best_algo, primary_metric].values[0]

        # HTML comparison table
        header_cols = list(comp_df.columns)
        header_html = "".join(
            f"<th style='background:#1A2B4A;color:#fff;padding:0.6rem 1rem;"
            f"text-align:left;font-size:0.85rem;'>{c}</th>"
            for c in header_cols
        )
        rows_html = ""
        for _, row in comp_df.iterrows():
            is_best = row["Algorithm"] == best_algo
            row_bg  = "#EAF5EE" if is_best else "#FFFFFF"
            row_fw  = "700"     if is_best else "400"
            cells   = ""
            for col in header_cols:
                badge = (
                    ' <span style="background:#2A7A4B;color:#fff;font-size:0.68rem;font-weight:700;'
                    'padding:0.1rem 0.4rem;border-radius:3px;letter-spacing:0.05em;">BEST</span>'
                    if col == "Algorithm" and is_best else ""
                )
                cells += (
                    f"<td style='padding:0.55rem 1rem;border-bottom:1px solid #E8E4DC;"
                    f"color:#111;font-weight:{row_fw};background:{row_bg};'>"
                    f"{row[col]}{badge}</td>"
                )
            rows_html += f"<tr>{cells}</tr>"

        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;font-family:sans-serif;
                      font-size:0.87rem;margin-bottom:0.8rem;'>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="success-box">
            <strong>Recommendation:</strong> <strong>{best_algo}</strong> achieved the highest
            {primary_metric} of <strong>{best_val}</strong> on the 20% held-out test set.
            Select this algorithm above and click <em>Train Model</em> to use it.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Algorithm Performance — Visual Comparison**")
        bar_colors = [PALETTE[0] if a == best_algo else PALETTE[3] for a in comp_df["Algorithm"]]
        fig, ax = styled_fig((9, 4))
        bars = ax.bar(comp_df["Algorithm"], comp_df[primary_metric],
                      color=bar_colors, edgecolor="none", width=0.5)
        for bar, val in zip(bars, comp_df[primary_metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#111")
        style_ax(ax, f"{primary_metric} by Algorithm", "Algorithm", primary_metric)
        ax.set_ylim(0, comp_df[primary_metric].max() * 1.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption(f"Taller bars indicate better performance. The dark bar highlights the best-performing algorithm on the {primary_metric} metric.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: SHAP ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Analysis":
    st.markdown('<div class="section-header">SHAP Analysis</div>', unsafe_allow_html=True)

    if "model_result" not in st.session_state:
        st.warning("Please train the model before running SHAP analysis.")
        st.stop()

    model, scaler, X_train, X_test, y_test, y_pred, metrics, feature_names, is_clf = st.session_state["model_result"]

    st.markdown("""
    <p style='font-size:0.9rem; color:#444; line-height:1.6;'>
    SHAP (SHapley Additive exPlanations) shows exactly how each feature pushes the
    model's prediction up or down. This makes the model's decision-making transparent
    and actionable for business users.
    </p>
    """, unsafe_allow_html=True)

    try:
        import shap

        with st.spinner("Computing SHAP values..."):
            X_test_df   = pd.DataFrame(X_test, columns=feature_names)
            X_train_df  = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
            X_test_sc   = pd.DataFrame(scaler.transform(X_test),  columns=feature_names)
            explainer   = shap.Explainer(model, X_train_df)
            shap_values = explainer(X_test_sc, check_additivity=False)

        # ── SHAP 1: Bar Summary ───────────────────────────────────────────────
        st.markdown("**Global Feature Impact — Mean Absolute SHAP Values**")
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_df   = pd.Series(mean_shap, index=feature_names).sort_values(ascending=False).head(15)

        fig, ax = styled_fig((10, 5))
        colors_shap = [PALETTE[0] if v == shap_df.max() else PALETTE[1] for v in shap_df.values]
        ax.barh(shap_df.index[::-1], shap_df.values[::-1], color=colors_shap[::-1],
                edgecolor="none", height=0.6)
        style_ax(ax, "Feature Impact on Predictions (Top 15)", "Mean |SHAP Value|", "Feature")
        ax.set_xlim(0, shap_df.max() * 1.15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Each bar shows the average impact of that feature on the model's output, across all records. The feature at the top has the largest influence on the predicted value. This view treats all records equally and provides the overall driver ranking.")

        # ── SHAP 2: Beeswarm (via matplotlib scatter proxy) ───────────────────
        st.markdown("<br>**SHAP Value Spread Across Records (Top 10 Features)**")
        top10 = shap_df.head(10).index.tolist()
        idx   = [feature_names.index(f) for f in top10]
        sv    = shap_values.values[:, idx]
        fv    = X_test_sc[top10].values

        fig, ax = styled_fig((11, 6))
        for i, feat in enumerate(top10[::-1]):
            col_idx = len(top10) - 1 - i
            s_vals  = sv[:, col_idx]
            f_vals  = fv[:, col_idx]
            norm_fv = (f_vals - f_vals.min()) / (f_vals.ptp() + 1e-9)
            colors_s = plt.cm.RdBu_r(norm_fv)
            jitter   = np.random.uniform(-0.2, 0.2, len(s_vals))
            ax.scatter(s_vals, np.full_like(s_vals, i) + jitter,
                       c=colors_s, alpha=0.5, s=12, linewidths=0)

        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10[::-1], fontsize=9)
        ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
        style_ax(ax, "SHAP Value Distribution Per Feature (Beeswarm)", "SHAP Value (Impact on Prediction)", "Feature")
        ax.grid(axis="x", color="#E8E4DC", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)

        # Colour legend
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cb.set_label("Feature Value (low → high)", fontsize=8.5, color="#333")
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Each dot represents one visit record. Dots to the right of the centre line increase the prediction; dots to the left decrease it. The colour indicates whether the feature value was low (blue) or high (red) for that record, revealing whether high or low values of each feature tend to drive sales up or down.")

        # ── SHAP 3: Dependence Plot for top feature ───────────────────────────
        top_feat    = shap_df.index[0]
        top_feat_i  = feature_names.index(top_feat)
        feat_vals   = X_test_sc[top_feat].values
        shap_top    = shap_values.values[:, top_feat_i]

        st.markdown(f"<br>**Dependency Plot — {top_feat}**")
        fig, ax = styled_fig((10, 4.5))
        sc = ax.scatter(feat_vals, shap_top, c=feat_vals, cmap="Blues",
                        alpha=0.65, s=18, edgecolors="none")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        z = np.polyfit(feat_vals, shap_top, 1)
        xline = np.linspace(feat_vals.min(), feat_vals.max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), color=PALETTE[2], linewidth=1.8, linestyle="-")
        style_ax(ax, f"SHAP Dependency: Effect of {top_feat} on Prediction",
                 f"{top_feat} (Scaled)", "SHAP Value (Contribution to Prediction)")
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(top_feat, fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption(f"This scatter plot shows how {top_feat} — the most impactful feature — affects predictions across all records. Each dot is one visit. The trend line (orange) shows the overall direction: whether higher values of {top_feat} tend to increase or decrease the predicted outcome.")

        # ── SHAP 4: Waterfall for single record ───────────────────────────────
        st.markdown("<br>**Single Record Explanation (First Test Record)**")
        rec_idx = 0
        sv_rec  = shap_values.values[rec_idx]
        base    = float(shap_values.base_values[rec_idx]) if hasattr(shap_values, "base_values") else 0.0

        top_n    = 12
        order    = np.argsort(np.abs(sv_rec))[::-1][:top_n]
        feats_wf = [feature_names[i] for i in order]
        vals_wf  = sv_rec[order]
        colors_w = [PALETTE[0] if v > 0 else "#B03A2E" for v in vals_wf]

        fig, ax = styled_fig((10, 5.5))
        ax.barh(feats_wf[::-1], vals_wf[::-1], color=colors_w[::-1],
                edgecolor="none", height=0.58)
        ax.axvline(0, color="#555", linewidth=0.9)
        style_ax(ax, "SHAP Waterfall — Top Contributing Features (Record 1)",
                 "SHAP Value (Impact on Prediction)", "Feature")
        ax.set_xlim(min(vals_wf.min() * 1.15, -0.01), vals_wf.max() * 1.15)

        pos_patch = mpatches.Patch(color=PALETTE[0], label="Increases prediction")
        neg_patch = mpatches.Patch(color="#B03A2E",  label="Decreases prediction")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption(f"This waterfall chart explains the prediction for one specific visit record. Blue bars represent features that pushed the prediction higher; red bars pushed it lower. The base value (model average) was {base:.2f}. This view helps answer: 'Why did the model predict this particular outcome for this record?'")

    except ImportError:
        st.markdown("""
        <div class="warn-box">
        <strong>SHAP library not installed.</strong><br>
        Run <code>pip install shap</code> in your environment to enable SHAP visualisations.
        The model has been trained successfully and all other sections remain functional.
        </div>
        """, unsafe_allow_html=True)

        # Fallback: show built-in feature importance
        st.markdown("**Feature Importance (Built-in) — Available Without SHAP**")
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
        fig, ax = styled_fig((10, 5))
        ax.barh(fi.index[::-1], fi.values[::-1], color=PALETTE[1], edgecolor="none", height=0.6)
        style_ax(ax, "Feature Importance (Top 15)", "Importance Score", "Feature")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("This chart shows the model's internal feature importance scores — a proxy for SHAP when the SHAP library is unavailable. Install SHAP for richer, per-prediction explanations.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: PERFORMANCE EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Performance Evaluation":
    st.markdown('<div class="section-header">Performance Evaluation</div>', unsafe_allow_html=True)

    if "model_result" not in st.session_state:
        st.warning("Please train the model first.")
        st.stop()

    model, scaler, X_train, X_test, y_test, y_pred, metrics, feature_names, is_clf = st.session_state["model_result"]
    target = st.session_state.get("target", "Target")

    st.markdown(f"""
    <p style='font-size:0.9rem; color:#444; line-height:1.6;'>
    These charts evaluate how well the trained model performs on the 20% held-out test set.
    The test set was never used during training, so these results reflect real-world predictive performance.
    </p>
    """, unsafe_allow_html=True)

    # ── Metric Summary ───────────────────────────────────────────────────────
    cols_m = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        kpi_metric(cols_m[i], k, v)

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    if not is_clf:
        # ── Regression: Actual vs Predicted ─────────────────────────────────
        st.markdown("<br>**Actual vs Predicted Values**")
        fig, ax = styled_fig((9, 5))
        ax.scatter(y_test_arr, y_pred_arr, alpha=0.45, color=PALETTE[1], s=22, edgecolors="none")
        lim_min = min(y_test_arr.min(), y_pred_arr.min())
        lim_max = max(y_test_arr.max(), y_pred_arr.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color=PALETTE[2],
                linewidth=1.8, linestyle="--", label="Perfect Prediction")
        style_ax(ax, f"Actual vs Predicted — {target}",
                 f"Actual {target}", f"Predicted {target}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Each dot represents one record from the test set. Dots close to the dashed line mean the model's prediction closely matched the actual value. The tighter the cluster around the line, the better the model's accuracy.")

        # ── Residuals Distribution ──────────────────────────────────────────
        st.markdown("<br>**Residuals Distribution**")
        residuals = y_test_arr - y_pred_arr
        fig, ax = styled_fig((9, 4.5))
        ax.hist(residuals, bins=35, color=PALETTE[0], edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.axvline(0,              color=PALETTE[2], linewidth=2,   linestyle="--", label="Zero Error")
        ax.axvline(np.mean(residuals), color="#B03A2E", linewidth=1.5, linestyle=":", label=f"Mean Error: {np.mean(residuals):.2f}")
        style_ax(ax, "Residual Error Distribution", "Prediction Error (Actual − Predicted)", "Frequency")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("Residuals are the differences between actual and predicted values. A well-performing model produces residuals centred near zero with a roughly symmetric spread. A skewed distribution or large mean error indicates the model tends to over- or under-predict systematically.")

        # ── Residuals vs Predicted ──────────────────────────────────────────
        st.markdown("<br>**Residuals vs Predicted (Error Pattern Check)**")
        fig, ax = styled_fig((9, 4.5))
        ax.scatter(y_pred_arr, residuals, alpha=0.4, color=PALETTE[1], s=18, edgecolors="none")
        ax.axhline(0, color=PALETTE[2], linewidth=1.5, linestyle="--")
        style_ax(ax, "Residuals vs Predicted Values",
                 f"Predicted {target}", "Residual (Actual − Predicted)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("This diagnostic plot reveals whether prediction errors change across different predicted value ranges. A random scatter around the zero line is ideal. A fan shape or trend in the residuals may indicate the model is less reliable for certain value ranges.")

        # ── Cumulative Error ─────────────────────────────────────────────────
        st.markdown("<br>**Cumulative Absolute Error**")
        sorted_abs_err = np.sort(np.abs(residuals))
        cumulative     = np.cumsum(sorted_abs_err) / np.sum(sorted_abs_err)
        fig, ax = styled_fig((9, 4))
        ax.plot(np.linspace(0, 100, len(cumulative)), cumulative * 100,
                color=PALETTE[0], linewidth=2)
        ax.axhline(80, color=PALETTE[2], linewidth=1.2, linestyle="--", label="80% threshold")
        style_ax(ax, "Cumulative Absolute Error Distribution",
                 "Percentage of Records (%)", "Cumulative Error Contribution (%)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("This chart shows what proportion of the total prediction error comes from what fraction of records. A steep early rise means a small number of records account for most of the error — indicating outliers or specific segments where the model struggles.")

    else:
        # ── Classification: Confusion Matrix ────────────────────────────────
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        st.markdown("<br>**Confusion Matrix**")
        labels = np.unique(np.concatenate([y_test_arr, y_pred_arr]))
        cm     = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#FFFFFF")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor="#E8E4DC",
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 10, "weight": "bold", "color": "#111"})
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", color="#111", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=10, color="#333")
        ax.set_ylabel("Actual Label",    fontsize=10, color="#333")
        ax.tick_params(labelsize=8.5, colors="#333")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_caption("The confusion matrix shows how often the model correctly classified each category (diagonal cells) versus how often it confused categories (off-diagonal cells). Large diagonal values and small off-diagonal values indicate a well-performing classifier.")

        # ── Per-Class Metrics ─────────────────────────────────────────────────
        from sklearn.metrics import classification_report
        st.markdown("<br>**Per-Class Performance Report**")
        report = classification_report(y_test_arr, y_pred_arr, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)
        chart_caption("This table breaks performance down by each class. Precision measures how often the model is correct when it predicts a class. Recall measures how many actual instances of that class were correctly identified. F1 balances the two.")

    # ── Export Results ────────────────────────────────────────────────────────
    st.markdown("<br>**Export Evaluation Results**")
    results_dict = {
        "Metric": list(metrics.keys()),
        "Value":  list(metrics.values()),
    }
    results_df  = pd.DataFrame(results_dict)
    csv_bytes   = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Metrics as CSV",
        data=csv_bytes,
        file_name="model_performance_metrics.csv",
        mime="text/csv",
    )
    chart_caption("Use the download button to export the performance metrics for reporting or record-keeping.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: STORE PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Store Performance":
    st.markdown('<div class="section-header">Store Performance Prediction</div>', unsafe_allow_html=True)

    if "df_raw" not in st.session_state:
        st.warning("Please upload a dataset first.")
        st.stop()
    if "model_result" not in st.session_state:
        st.markdown("""
        <div class="warn-box">
        <strong>Model not trained yet.</strong><br>
        A trained model is required to generate store-level predictions.
        Please complete <strong>Preprocessing → Model Training</strong> before using this page.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df_raw  = st.session_state["df_raw"]
    df_proc = st.session_state.get("df_processed", None)
    model, scaler, X_train, X_test, y_test, y_pred, metrics, feature_names, is_clf = st.session_state["model_result"]
    target  = st.session_state.get("target", "SalesQty")

    st.markdown(f"""
    <p style='font-size:0.9rem; color:#111111; font-weight:600; line-height:1.7;'>
    Select a specific store (outlet) from your dataset and adjust key performance parameters
    to simulate how changes in those factors would affect the predicted
    <strong>{target}</strong>.
    This tool helps identify practical levers for improving individual store performance.
    </p>
    """, unsafe_allow_html=True)

    # ── Store selector ────────────────────────────────────────────────────────
    outlet_col = None
    for candidate in ["OutletName", "OutletID", "DistributorCode"]:
        if candidate in df_raw.columns:
            outlet_col = candidate
            break

    if outlet_col is None:
        st.warning("No outlet identifier column found (expected OutletName, OutletID, or DistributorCode).")
        st.stop()

    outlet_list = sorted(df_raw[outlet_col].dropna().astype(str).unique().tolist())
    if len(outlet_list) == 0:
        st.warning("No store records found in the dataset.")
        st.stop()

    selected_store = st.selectbox(f"🏪 Select Store ({outlet_col})", options=outlet_list)

    store_mask   = df_raw[outlet_col].astype(str) == selected_store
    store_raw_df = df_raw[store_mask].copy()

    if df_proc is not None and len(store_raw_df) > 0:
        try:
            store_proc = run_preprocessing(store_raw_df)
            for col in feature_names:
                if col not in store_proc.columns:
                    store_proc[col] = 0
            store_proc = store_proc[feature_names]
            base_row = store_proc.iloc[0].copy()
        except Exception:
            base_row = pd.Series(
                {f: float(df_proc[f].mean()) for f in feature_names if f in df_proc.columns}
            )
            for f in feature_names:
                if f not in base_row.index:
                    base_row[f] = 0.0
    else:
        base_row = pd.Series({f: 0.0 for f in feature_names})

    # ── Store Summary KPI cards ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:1rem; font-weight:800; color:#111111;
                border-bottom:2px solid #C8943A; padding-bottom:0.3rem;
                margin-bottom:0.9rem;'>
        Store Summary — {selected_store}
    </div>
    """, unsafe_allow_html=True)

    total_visits    = len(store_raw_df)
    total_sales_qty = (pd.to_numeric(store_raw_df["SalesQty"], errors="coerce").sum()
                       if "SalesQty" in store_raw_df.columns else None)
    total_sales_val = (pd.to_numeric(store_raw_df["SalesValue"], errors="coerce").sum()
                       if "SalesValue" in store_raw_df.columns else None)
    avg_dur         = (pd.to_numeric(store_raw_df["CallDuration"], errors="coerce").mean()
                       if "CallDuration" in store_raw_df.columns else None)

    def kpi_card(col, icon, label, value):
        col.markdown(f"""
        <div style='background:#FFFFFF; border:2px solid #1A2B4A; border-radius:8px;
                    padding:1.1rem 1rem; box-shadow:0 4px 14px rgba(26,43,74,0.12);
                    text-align:center;'>
            <div style='font-size:1.3rem; margin-bottom:0.2rem;'>{icon}</div>
            <div style='font-size:0.72rem; font-weight:800; letter-spacing:0.08em;
                        text-transform:uppercase; color:#1A2B4A; margin-bottom:0.35rem;'>
                {label}
            </div>
            <div style='font-size:1.6rem; font-weight:900; color:#000000; line-height:1;'>
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    kc1, kc2, kc3, kc4 = st.columns(4)
    kpi_card(kc1, "🏪", "Total Visits", f"{total_visits:,}")
    kpi_card(kc2, "📦", "Total Sales Qty",
             f"{int(total_sales_qty):,}" if total_sales_qty is not None else "N/A")
    kpi_card(kc3, "💰", "Total Sales Value",
             f"{total_sales_val:,.0f}" if total_sales_val is not None else "N/A")
    kpi_card(kc4, "⏱️", "Avg Call Duration",
             f"{avg_dur:.1f} min" if avg_dur is not None else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#D8D4CC;'>", unsafe_allow_html=True)

    # ── Parameter Adjustment Panel ────────────────────────────────────────────
    st.markdown(f"""
    <div style='font-size:1rem; font-weight:800; color:#111111;
                border-bottom:2px solid #C8943A; padding-bottom:0.3rem;
                margin-bottom:0.6rem;'>
        🎛️ Adjust Parameters to Simulate Performance
    </div>
    <p style='font-size:0.88rem; color:#111111; font-weight:600; line-height:1.7; margin-bottom:1rem;'>
    Use the sliders below to change key operational parameters for this store.
    The model will re-predict the expected <strong>{target}</strong> based on your adjustments.
    Compare the <em>Baseline</em> (current values) against the <em>Simulated</em> scenario.
    </p>
    """, unsafe_allow_html=True)

    ADJUSTABLE_MAP = {
        "CallDuration":    ("Call Duration (minutes)",          0.0,  120.0, 1.0),
        "SalesQty":        ("Sales Quantity (units)",           0.0,  500.0, 1.0),
        "DeliveryQty":     ("Delivery Quantity (units)",        0.0,  500.0, 1.0),
        "DeliveryDelay":   ("Delivery Delay (days)",            0.0,   30.0, 0.5),
        "GeoDeviation_km": ("GPS Deviation from Outlet (km)",  0.0,   10.0, 0.1),
        "VisitSuccess":    ("Visit Success Flag (0=No, 1=Yes)", 0.0,    1.0, 1.0),
        "Month":           ("Month of Visit (1–12)",            1.0,   12.0, 1.0),
        "DayOfWeek":       ("Day of Week (0=Mon … 6=Sun)",      0.0,    6.0, 1.0),
        "WeekNum":         ("Week Number (1–52)",               1.0,   52.0, 1.0),
    }

    adjustable_features = [f for f in feature_names if f in ADJUSTABLE_MAP]

    if not adjustable_features:
        st.info("No recognised adjustable parameters found in this model's feature set. "
                "Ensure preprocessing has been run with the standard column schema.")
        st.stop()

    sim_row   = base_row.copy()
    left_col, right_col = st.columns(2)
    for i, feat in enumerate(adjustable_features):
        label, mn, mx, step = ADJUSTABLE_MAP[feat]
        current_val = float(base_row.get(feat, (mn + mx) / 2))
        current_val = max(mn, min(mx, current_val))
        col = left_col if i % 2 == 0 else right_col
        with col:
            st.markdown(
                f"<p style='font-size:0.83rem;font-weight:700;color:#111111;"
                f"margin-bottom:0;margin-top:0.4rem;'>{label}</p>",
                unsafe_allow_html=True,
            )
            new_val = st.slider(
                label=label, label_visibility="collapsed",
                min_value=float(mn), max_value=float(mx),
                value=current_val, step=float(step),
                key=f"slider_{feat}",
            )
            sim_row[feat] = new_val

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_row(row: pd.Series) -> float:
        arr        = np.array([[row[f] for f in feature_names]])
        arr_scaled = scaler.transform(arr)
        return float(model.predict(arr_scaled)[0])

    baseline_pred  = predict_row(base_row)
    simulated_pred = predict_row(sim_row)
    delta          = simulated_pred - baseline_pred
    delta_pct      = (delta / abs(baseline_pred) * 100) if baseline_pred != 0 else 0.0

    # ── Prediction Results ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#D8D4CC;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:1rem; font-weight:800; color:#111111;
                border-bottom:2px solid #C8943A; padding-bottom:0.3rem;
                margin-bottom:0.9rem;'>
        📈 Prediction Results
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    arrow   = "↑" if delta > 0 else ("→" if delta == 0 else "↓")
    dir_lbl = "Gain" if delta > 0 else ("Flat" if delta == 0 else "Loss")
    card_bg_sim  = "#EAF5EE" if delta >= 0 else "#FDF3E7"
    border_sim   = "#2A7A4B" if delta >= 0 else "#C8943A"
    fg_sim       = "#1A4A30" if delta >= 0 else "#6B3C00"

    for col, icon, label, val_str, bg, border, fg in [
        (rc1, "📊", f"Baseline {target}",   f"{baseline_pred:,.2f}",  "#EEF2F8", "#2E5090", "#1A2B4A"),
        (rc2, "🎯", f"Simulated {target}",  f"{simulated_pred:,.2f}", card_bg_sim, border_sim, fg_sim),
        (rc3, arrow, "Change Direction",
         f"{arrow} {dir_lbl}  ({delta:+.2f}, {delta_pct:+.1f}%)",
         card_bg_sim, border_sim, fg_sim),
    ]:
        col.markdown(f"""
        <div style='background:{bg}; border:2px solid {border}; border-radius:8px;
                    padding:1rem 1.1rem; text-align:center;'>
            <div style='font-size:1.2rem; margin-bottom:0.2rem;'>{icon}</div>
            <div style='font-size:0.72rem; font-weight:800; letter-spacing:0.07em;
                        text-transform:uppercase; color:{fg}; margin-bottom:0.3rem;'>
                {label}
            </div>
            <div style='font-size:1.3rem; font-weight:900; color:{fg}; line-height:1.2;'>
                {val_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if delta > 0:
        box_class, interp = "success-box", (
            f"The adjusted parameters result in a predicted <strong>{target}</strong> of "
            f"<strong>{simulated_pred:,.2f}</strong>, an improvement of "
            f"<strong>{delta:+.2f} ({delta_pct:+.1f}%)</strong> over the baseline of "
            f"{baseline_pred:,.2f}. Consider applying these operational changes to this store.")
    elif delta < 0:
        box_class, interp = "warn-box", (
            f"The adjusted parameters result in a predicted <strong>{target}</strong> of "
            f"<strong>{simulated_pred:,.2f}</strong>, a decline of "
            f"<strong>{delta:.2f} ({delta_pct:.1f}%)</strong> vs the baseline of "
            f"{baseline_pred:,.2f}. Revert these changes or try different parameter combinations.")
    else:
        box_class, interp = "insight-box", (
            f"The adjusted parameters produce the same predicted <strong>{target}</strong> "
            f"({simulated_pred:,.2f}) as the current baseline. Try adjusting other parameters.")

    st.markdown(f'<div class="{box_class}">{interp}</div>', unsafe_allow_html=True)

    # ── Baseline vs Simulated bar chart ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = styled_fig((7, 3.5))
    bar_vals   = [baseline_pred, simulated_pred]
    bar_labels = ["Baseline", "Simulated"]
    bar_colors = [PALETTE[1], PALETTE[0] if delta >= 0 else "#B03A2E"]
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="none", width=0.4)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in bar_vals) * 0.02,
                f"{val:,.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#111111")
    style_ax(ax, f"Baseline vs Simulated — {target} for {selected_store}",
             "Scenario", f"Predicted {target}")
    ax.set_ylim(0, max(bar_vals) * 1.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    chart_caption(
        f"This bar chart compares the model's predicted {target} under current conditions (Baseline) "
        f"vs the scenario you configured using the sliders (Simulated). "
        f"Each bar represents one complete prediction from the trained model."
    )

    # ── Parameter change summary ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<strong style='font-size:0.95rem; color:#111111;'>Parameter Change Summary</strong>",
                unsafe_allow_html=True)
    change_rows = []
    for feat in adjustable_features:
        label, *_ = ADJUSTABLE_MAP[feat]
        orig = float(base_row.get(feat, 0))
        new  = float(sim_row[feat])
        diff = new - orig
        change_rows.append({
            "Parameter":       label,
            "Current Value":   f"{orig:.2f}",
            "Simulated Value": f"{new:.2f}",
            "Change":          f"{diff:+.2f}",
        })
    st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
    chart_caption("This table summarises all parameter adjustments made in this simulation. "
                  "Share with field teams to communicate the operational changes being recommended.")



# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "AI Assistant":

    # ── API Key — INSERT YOUR KEY BELOW ──────────────────────────────────────
    # Replace the empty string with your Google Gemini API key, e.g.:
    #   GEMINI_API_KEY = "AIzaSy..."
    GEMINI_API_KEY = "AIzaSyDCFsO0-1H-m4vturYyTzwSSyq3voakorQ"   # <-- INSERT YOUR GEMINI API KEY HERE
    # ─────────────────────────────────────────────────────────────────────────
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") 

    # ── Model auto-discovery — finds what's actually available with YOUR key ──
    # No hardcoded model names. Queries Google's own list endpoint at runtime.
    GEMINI_PREFERRED = ["flash", "pro"]   # preference order within available models

    import requests
    import json as _json
    import re as _re
    import time as _time

    @st.cache_data(ttl=3600, show_spinner=False)
    def _discover_gemini_model(api_key: str) -> str:
        """
        Query the Gemini models endpoint with the real API key and return the
        best available generative model name. Result cached for 1 hour.
        """
        try:
            r = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=10,
            )
            if r.status_code != 200:
                return None
            models = r.json().get("models", [])
            # Keep only models that support generateContent
            generative = [
                m["name"].replace("models/", "")
                for m in models
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            # Prefer flash variants (faster/cheaper), then pro
            for keyword in ["flash", "pro", "gemini"]:
                matches = [m for m in generative if keyword in m.lower()]
                if matches:
                    # Pick the newest-looking one (highest version number heuristic)
                    matches.sort(reverse=True)
                    return matches[0]
            return generative[0] if generative else None
        except Exception:
            return None

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.92rem; color:#444; line-height:1.7; max-width:760px;'>
    Your intelligent analytics companion. Ask anything about your sales data, model
    performance, SHAP results, or distribution strategy — and get clear, plain-English
    answers tailored to what's loaded in this session.
    </p>
    """, unsafe_allow_html=True)

    # ── Gate: API key missing ─────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        st.markdown("""
        <div class="warn-box" style='max-width:680px;'>
            <strong>🔑 Gemini API Key Required</strong><br><br>
            To activate the AI Assistant:<br>
            1. Open <code>shap_dashboard.py</code> in any text editor.<br>
            2. Find the line &nbsp;<code>GEMINI_API_KEY = ""</code>&nbsp; near the top
               of the <em>AI Assistant</em> section.<br>
            3. Paste your key inside the quotes and save the file.<br>
            4. Restart the app — the assistant will be live immediately.<br><br>
            Don't have a key yet?
            Get a free one at <strong>aistudio.google.com/app/apikey</strong>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def build_system_prompt() -> str:
        """Build a rich system prompt from whatever is loaded in session state."""
        lines = [
            "You are a friendly, expert sales analytics assistant embedded inside a "
            "professional SHAP-based distributor sales dashboard.",
            "Your role: help business users understand their data, model results, and "
            "SHAP explanations in plain, jargon-free English.",
            "Always be concise, warm, and actionable. Use bullet points only when listing "
            "multiple items. Format numbers clearly. Never mention code or technical "
            "implementation details unless the user specifically asks.",
        ]
        ctx = []
        if "df_raw" in st.session_state:
            df_r = st.session_state["df_raw"]
            missing = round(df_r.isnull().mean().mean() * 100, 1)
            ctx.append(
                f"Raw dataset: {len(df_r):,} records, {len(df_r.columns)} columns, "
                f"{missing}% missing values."
            )
        if "df_processed" in st.session_state:
            df_p = st.session_state["df_processed"]
            ctx.append(
                f"Processed dataset: {len(df_p):,} records, "
                f"{len(df_p.columns)} model-ready features."
            )
        if "model_result" in st.session_state:
            _, _, _, _, _, _, mets, feats, is_clf = st.session_state["model_result"]
            tgt  = st.session_state.get("target", "unknown")
            algo = st.session_state.get("trained_algo", "unknown")
            task = "classification" if is_clf else "regression"
            mstr = ", ".join(f"{k}: {v}" for k, v in mets.items())
            ctx.append(
                f"Trained model: {algo} ({task}), predicting '{tgt}'. "
                f"Performance — {mstr}. "
                f"Top 5 features: {', '.join(feats[:5])}."
            )
        if ctx:
            lines.append("Current dashboard context: " + " | ".join(ctx))
        return "\n".join(lines)

    def call_gemini(history: list, system_prompt: str) -> str:
        """
        Discover the best available Gemini model for this API key, then call it.
        Retries up to 3× on 429 with exponential back-off.
        """
        # ── Discover model ────────────────────────────────────────────────────
        model_name = _discover_gemini_model(GEMINI_API_KEY)
        if not model_name:
            # Discovery failed — fall back to the most universally available name
            model_name = "gemini-pro"

        # ── Build request ─────────────────────────────────────────────────────
        contents = []
        for m in history:
            contents.append({
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            })

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature":     0.4,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        # ── Call with retry on 429 ────────────────────────────────────────────
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=_json.dumps(payload),
                    timeout=90,
                )
            except requests.exceptions.Timeout:
                raise ValueError("The request timed out. Please try again.")
            except Exception as exc:
                raise ValueError(f"Network error: {exc}")

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
                raise ValueError(
                    "The assistant couldn't generate a response. "
                    "Please try rephrasing your question."
                )

            elif resp.status_code == 429:
                if attempt < 2:
                    _time.sleep(6 * (2 ** attempt))   # 6 s → 12 s
                    continue
                raise ValueError(
                    "The AI service is temporarily busy. "
                    "Please wait 30 seconds and try again."
                )

            elif resp.status_code == 403:
                raise ValueError(
                    "Access denied — your Gemini API key is invalid or expired.\n"
                    "Get a valid key at aistudio.google.com/app/apikey"
                )

            elif resp.status_code == 404:
                # Cached model name stale — clear cache and raise so user retries
                _discover_gemini_model.clear()
                raise ValueError(
                    "The AI model is temporarily unavailable. "
                    "Please try again in a moment."
                )

            else:
                raise ValueError(
                    f"Something went wrong (error {resp.status_code}). "
                    "Please try again."
                )

        raise ValueError("Failed after 3 attempts. Please try again shortly.")

    def render_message(role: str, content: str):
        """Render a single chat bubble with proper markdown-to-HTML conversion."""
        is_user  = (role == "user")
        label    = "You" if is_user else "🤖 AI Assistant"
        bg       = "#EEF3FB" if is_user else "#F2FAF4"
        border   = "#2E5090" if is_user else "#2A7A4B"
        align    = "flex-end" if is_user else "flex-start"
        radius   = "12px 4px 12px 12px" if is_user else "4px 12px 12px 12px"
        lbl_col  = "#2E5090" if is_user else "#2A7A4B"

        # Convert markdown bold/italic to HTML for display inside st.markdown HTML block
        html_content = content
        html_content = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = _re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         html_content)
        # Convert bullet lines to HTML list items
        lines = html_content.split("\n")
        out, in_list = [], False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith(("- ", "• ", "* ")):
                if not in_list:
                    out.append("<ul style='margin:0.4rem 0 0.4rem 1.1rem; padding:0;'>")
                    in_list = True
                out.append(f"<li style='margin-bottom:0.2rem;'>{stripped[2:]}</li>")
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                if stripped:
                    out.append(f"<p style='margin:0.3rem 0;'>{stripped}</p>")
        if in_list:
            out.append("</ul>")
        html_content = "".join(out)

        st.markdown(f"""
        <div style='display:flex; justify-content:{align}; margin-bottom:1rem;'>
            <div style='max-width:78%; background:{bg}; border:1px solid {border}22;
                        border-radius:{radius}; padding:0.85rem 1.1rem;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='font-size:0.68rem; font-weight:800; letter-spacing:0.09em;
                            text-transform:uppercase; color:{lbl_col};
                            margin-bottom:0.45rem;'>{label}</div>
                <div style='font-size:0.9rem; color:#1A1A1A; line-height:1.65;'>
                    {html_content}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Session state init ────────────────────────────────────────────────────
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_pending_input"  not in st.session_state:
        st.session_state["ai_pending_input"] = ""
    if "ai_error_msg"      not in st.session_state:
        st.session_state["ai_error_msg"] = ""

    # ── Quick-start prompt chips ──────────────────────────────────────────────
    if not st.session_state["ai_chat_history"]:
        st.markdown("""
        <div style='margin-bottom:1.2rem;'>
            <div style='font-size:0.78rem; font-weight:700; letter-spacing:0.06em;
                        text-transform:uppercase; color:#888; margin-bottom:0.6rem;'>
                Quick questions to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
        chips = [
            "Summarise the model performance",
            "Which features drive sales the most?",
            "How do I interpret SHAP values?",
            "What do the missing values mean for my analysis?",
            "Which stores or regions need the most attention?",
        ]
        chip_cols = st.columns(len(chips))
        for ci, (chip_col, chip_text) in enumerate(zip(chip_cols, chips)):
            if chip_col.button(chip_text, key=f"chip_{ci}", use_container_width=True):
                st.session_state["ai_pending_input"] = chip_text

    # ── Chat history display ──────────────────────────────────────────────────
    for msg in st.session_state["ai_chat_history"]:
        render_message(msg["role"], msg["content"])

    # ── Error display ─────────────────────────────────────────────────────────
    if st.session_state["ai_error_msg"]:
        st.markdown(f"""
        <div class="warn-box" style='margin-bottom:0.8rem;'>
            ⚠️ {st.session_state["ai_error_msg"]}
        </div>
        """, unsafe_allow_html=True)
        st.session_state["ai_error_msg"] = ""

    # ── Input area ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    col_inp, col_btn = st.columns([6, 1])
    with col_inp:
        user_input = st.text_input(
            "Message",
            value=st.session_state["ai_pending_input"],
            placeholder="Type your question here and press Send…",
            label_visibility="collapsed",
            key="ai_text_input",
        )
    with col_btn:
        send_clicked = st.button("Send ➤", use_container_width=True, key="ai_send")

    # Also allow Enter key (Streamlit fires on_change when input is submitted)
    enter_submitted = (
        user_input
        and user_input != st.session_state.get("ai_last_sent", "")
        and not send_clicked
    )

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([1, 5])
    with ctrl_col1:
        if st.session_state["ai_chat_history"]:
            if st.button("🗑 Clear chat", key="ai_clear", use_container_width=True):
                st.session_state["ai_chat_history"] = []
                st.session_state["ai_pending_input"] = ""
                st.session_state["ai_error_msg"]     = ""
                st.rerun()

    # ── Show which model is active (discovered at runtime) ───────────────────
    _active_model = _discover_gemini_model(GEMINI_API_KEY)
    _model_label  = _active_model if _active_model else "auto-detecting…"
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#999; margin-top:0.5rem; line-height:1.6;'>
        💡 <em>The assistant knows what data, model, and results are currently loaded
        in your session — no need to explain the context manually.</em><br>
        <span style='color:#bbb;'>Active AI model: <strong style='color:#aaa;'>{_model_label}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Process send ──────────────────────────────────────────────────────────
    trigger_text = user_input.strip() if (send_clicked or enter_submitted) else ""
    if not trigger_text and st.session_state["ai_pending_input"]:
        trigger_text = st.session_state["ai_pending_input"].strip()
        st.session_state["ai_pending_input"] = ""

    if trigger_text:
        st.session_state["ai_last_sent"] = trigger_text

        # Add user message to history first
        st.session_state["ai_chat_history"].append(
            {"role": "user", "content": trigger_text}
        )

        # Call Gemini
        with st.spinner("Thinking…"):
            try:
                system_prompt  = build_system_prompt()
                reply          = call_gemini(st.session_state["ai_chat_history"], system_prompt)
                st.session_state["ai_chat_history"].append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as exc:
                # Remove the user message we just added so the user can retry
                st.session_state["ai_chat_history"].pop()
                st.session_state["ai_error_msg"] = str(exc)

        st.rerun()
