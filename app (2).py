import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sheet Pile Service Life Predictor",
    page_icon="🏗️",
    layout="centered",
)

# ─── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "sheet_pile_rfr_model__7_.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    # pkl is saved as a dict with key 'model'
    if isinstance(data, dict) and "model" in data:
        return data["model"]
    return data

model = load_model()

# ─── USCS Soil Classification Encoding ─────────────────────────────────────────
# The model uses a label-encoded 'Soil_Type' column.
# Map each USCS class to its integer code (derived from training data alphabetical order).
USCS_LABEL_MAP = {
    "CH – Fat Clay":               0,
    "CL – Lean Clay":              1,
    "GC – Clayey Gravel":          2,
    "GM – Silty Gravel":           3,
    "GP – Poorly Graded Gravel":   4,
    "GW – Well Graded Gravel":     5,
    "MH – Elastic Silt":           6,
    "ML – Silt":                   7,
    "OL – Organic Clay (low pl.)": 8,
    "PT – Peat":                   9,
    "SC – Clayey Sand":            10,
    "SM – Silty Sand":             11,
    "SP – Poorly Graded Sand":     12,
    "SW – Well Graded Sand":       13,
}

USCS_SHORT = {v: k.split(" – ")[0] for k, v in USCS_LABEL_MAP.items()}

# ─── Baseline / Average Values for Hidden Features ─────────────────────────────
# These are domain-appropriate average baselines; they keep the model well-conditioned
# without requiring user input for less-sensitive geoenvironmental parameters.
BASELINE = {
    "Lateral Effective Stress (kPa)":          25.0,
    "Vertical Effective Stress (kPa)":         60.0,
    "SPT Corrected N-Values":                  15.0,
    "Internal Friction Angle (deg)":           30.0,
    "Effective Cohesion (kPa)":                10.0,
    "Saturated Unit Weight (kN/m³)":           18.5,
    "Mean Chloride Deposition":                 0.05,
    "Mean Sulfate Deposition":                  0.02,
    "Mean Humidity (%)":                        70.0,
    "Mean Annual Temp (°C)":                    25.0,
    "Void Ratio":                               0.65,
    "Porosity":                                 0.40,
}

# Exact column names as stored in the model's feature_names_in_
FEATURE_ORDER = [
    "Lateral Effective Stress (kPa)",
    "Vertical Effective Stress (kPa)",
    "Surcharge (kPa)",
    "Embedment Depth (m)",
    "(Groundwater Table Elev. from Surface (m)",   # exact name from pkl
    "SPT Corrected N-Values",
    "Internal Friction Angle (deg)",
    "Effective Cohesion (kPa)",
    "Saturated Unit Weight (kN/m\xb3)",            # ³ character
    "Factored Corrosion Rate (mm/yr)",
    "Flange Initial Thick. (mm)",
    "Mean Chloride Deposition",
    "Mean Sulfate Deposition",
    "Mean Humidity (%)",
    "Mean Annual Temp (\xb0C)",                    # °C
    "Void Ratio",
    "Porosity",
    "Soil_Type",
]

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("🏗️ Sheet Pile Service Life Predictor")
st.markdown(
    """
    This tool uses a **Random Forest Regression** model to estimate the 
    service life (in years) of sheet pile structures based on soil and 
    structural parameters.
    """,
    unsafe_allow_html=False,
)

st.divider()

# ── Section 1: Soil Classification ───────────────────────────────────────────
st.subheader("🪨 Soil Classification")
uscs_label = st.selectbox(
    "USCS Soil Classification",
    options=list(USCS_LABEL_MAP.keys()),
    index=13,   # default: SW – Well Graded Sand
    help="Select the Unified Soil Classification System (USCS) group symbol that best "
         "describes the soil at the sheet pile location.",
)
soil_code = USCS_LABEL_MAP[uscs_label]

st.divider()

# ── Section 2: Structural Parameters ─────────────────────────────────────────
st.subheader("🔩 Structural Parameters")

col1, col2 = st.columns(2)
with col1:
    flange_thick = st.slider(
        "Flange Initial Thickness (mm)",
        min_value=5.0, max_value=30.0, value=15.0, step=0.5,
        help="Initial thickness of the sheet pile flange in millimetres.",
    )
with col2:
    corrosion_rate = st.slider(
        "Factored Corrosion Rate (mm/yr)",
        min_value=0.01, max_value=1.00, value=0.10, step=0.01,
        help="Annual steel section loss rate after applying environmental factors.",
    )

st.divider()

# ── Section 3: Loading & Site Conditions ─────────────────────────────────────
st.subheader("📐 Loading & Site Conditions")

col3, col4 = st.columns(2)
with col3:
    surcharge = st.slider(
        "Surcharge (kPa)",
        min_value=0.0, max_value=100.0, value=10.0, step=1.0,
        help="Applied surface surcharge pressure behind the sheet pile wall.",
    )
    embedment = st.slider(
        "Embedment Depth (m)",
        min_value=1.0, max_value=20.0, value=6.0, step=0.5,
        help="Depth of sheet pile penetration below the excavation or dredge level.",
    )
with col4:
    gwt = st.slider(
        "Groundwater Table Elevation from Surface (m)",
        min_value=0.0, max_value=15.0, value=2.0, step=0.5,
        help="Depth to groundwater table measured from ground surface (positive = below surface).",
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Service Life", type="primary", use_container_width=True)

if predict_btn:
    # Build the input dict using baseline + user inputs
    input_dict = {
        # Hidden baseline features
        "Lateral Effective Stress (kPa)":           BASELINE["Lateral Effective Stress (kPa)"],
        "Vertical Effective Stress (kPa)":          BASELINE["Vertical Effective Stress (kPa)"],
        "Surcharge (kPa)":                          surcharge,
        "Embedment Depth (m)":                      embedment,
        "(Groundwater Table Elev. from Surface (m)": gwt,
        "SPT Corrected N-Values":                   BASELINE["SPT Corrected N-Values"],
        "Internal Friction Angle (deg)":            BASELINE["Internal Friction Angle (deg)"],
        "Effective Cohesion (kPa)":                 BASELINE["Effective Cohesion (kPa)"],
        "Saturated Unit Weight (kN/m\xb3)":         BASELINE["Saturated Unit Weight (kN/m³)"],
        "Factored Corrosion Rate (mm/yr)":          corrosion_rate,
        "Flange Initial Thick. (mm)":               flange_thick,
        "Mean Chloride Deposition":                 BASELINE["Mean Chloride Deposition"],
        "Mean Sulfate Deposition":                  BASELINE["Mean Sulfate Deposition"],
        "Mean Humidity (%)":                        BASELINE["Mean Humidity (%)"],
        "Mean Annual Temp (\xb0C)":                 BASELINE["Mean Annual Temp (°C)"],
        "Void Ratio":                               BASELINE["Void Ratio"],
        "Porosity":                                 BASELINE["Porosity"],
        "Soil_Type":                                soil_code,
    }

    # Build DataFrame in exact feature order that the model was trained on
    X = pd.DataFrame([input_dict])[FEATURE_ORDER]

    try:
        prediction = model.predict(X)[0]
        prediction = max(0.0, prediction)   # clamp to non-negative

        # Result card
        st.success("✅ Prediction complete!")
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.metric(
                label="Estimated Service Life",
                value=f"{prediction:.1f} years",
            )

        # Show input summary
        with st.expander("📋 Input Summary", expanded=False):
            summary = {
                "USCS Soil Type": uscs_label,
                "Soil_Type (encoded)": soil_code,
                "Flange Initial Thick. (mm)": flange_thick,
                "Factored Corrosion Rate (mm/yr)": corrosion_rate,
                "Surcharge (kPa)": surcharge,
                "Embedment Depth (m)": embedment,
                "GWT from Surface (m)": gwt,
                **{f"[baseline] {k}": v for k, v in BASELINE.items()},
            }
            st.dataframe(
                pd.DataFrame(summary.items(), columns=["Parameter", "Value"]),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure the model file `sheet_pile_rfr_model__7_.pkl` is in the same directory as `app.py`.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Random Forest Regressor (scikit-learn 1.6.1) · "
    "Baseline hidden features use average geoenvironmental values."
)
