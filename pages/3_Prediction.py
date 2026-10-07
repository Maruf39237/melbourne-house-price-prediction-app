import numpy as np
import pandas as pd
import streamlit as st
import datetime
import random
import joblib
import pickle
import ast
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path


BASE_DIR = Path.cwd()
MODEL_PATH      = BASE_DIR / "Model" / "linear_regression.pkl"
FEATURE_PATH    = BASE_DIR / "Model" / "feature_names.pkl"
RAW_DATA_PATH   = BASE_DIR / "Data" / "Melbourne_housing.csv"
CLEAN_DATA_PATH = BASE_DIR / "Data" / "Cleaned_dataset.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, feature_names

@st.cache_data
def load_data():
    raw = pd.read_csv(RAW_DATA_PATH, on_bad_lines="skip")
    clean = pd.read_csv(CLEAN_DATA_PATH, on_bad_lines="skip")
    return raw, clean

df1, df2 = load_data()
st.divider()
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div.stButton > button:first-child {
    background: linear-gradient(90deg,#ff9800,#ff5722);
    color:white;
    border:none;
    border-radius:12px;
    padding:12px 25px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
    }

    div.stButton > button:first-child:hover{
        transform:scale(1.03);
        box-shadow:0px 5px 18px rgba(0,0,0,.25);
    }
    .prediction-container {
        background: linear-gradient(135deg, 
            rgba(34, 197, 94, 0.12) 0%, 
            rgba(22, 163, 74, 0.05) 100%
        ) !important;
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 15px;
    }
    
    .prediction-title {
        color: var(--text-color) !important;
        opacity: 100;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    
    .prediction-value {
        color: #10b981 !important; 
        margin: 0;
        font-size: 32px;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("## Melbourne House Price Prediction System🏠")
    st.markdown("""##### Enter the property's characteristics to estimate its market value. The prediction is generated using a trained Linear Regression model built on the Melbourne Housing dataset.""")

with col2:
    url = "https://images.pexels.com/photos/20208884/pexels-photo-20208884.jpeg"
    st.image(url, width=360, caption="ohooo it's prediction time!!! 🥳")

st.divider()

with st.expander("🗂️ Click to view raw data"):
    st.dataframe(df1.head(), use_container_width=True)

with st.expander("📊 Click to view cleaned data"):
    st.dataframe(df2.head(), use_container_width=True)

with st.expander("🔗 Click to view the pipeline behind the scene"):
    st.write(""" 
    ```text 
        The pipeline behind the scene:

                    User Input
                        ↓
                    Validation
                        ↓
        Same preprocessing used in notebook
                        ↓
                One Hot Encoding
                        ↓
                Column Alignment
                        ↓
                    Prediction
                        ↓
            Inverse Log Transformation
                        ↓
                Predicted Price
    ```""")

st.divider()

def cast_and_store_state(data_dict):
    int_keys = ["Rooms", "Bedroom", "Bathroom", "Car", "Propertycount", "Landsize", "BuildingArea", "YearBuilt"]
    float_keys = ["Distance", "Postcode", "Latitude", "Longitude"]
    str_keys = ["Type", "Method", "Regionname", "ParkingArea"]
    
    for k, v in data_dict.items():
        try:
            if k in int_keys:
                st.session_state[k] = int(float(v))
            elif k in float_keys:
                st.session_state[k] = float(v)
            elif k in str_keys:
                val_str = str(v)
                if k in df1.columns and val_str in df1[k].dropna().unique().tolist():
                    st.session_state[k] = val_str
        except Exception:
            pass
    st.toast("⚡ Loaded variables imported seamlessly!", icon="🔄")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✍️ Manual Entry",
    "📋 Paste Dictionary",
    "📄 Paste CSV",
    "🎲 Load Random House",
    "🔢 Load Dataset Row"
])


with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏠 Property")

        Rooms = st.number_input("Rooms", min_value=1, value=4, key="Rooms")
        Bedrooms = st.number_input("Bedroom", min_value=1, value=3, key="Bedroom")
        Bathrooms = st.number_input("Bathroom", min_value=1, value=2, key="Bathroom")
        # Car = st.number_input("Car", min_value=0, value=1, key="Car")
        Car = st.number_input("Car", min_value=0, value=1, key="Car")
        Property = st.number_input("PropertyCount", min_value=1, value=4030, key="Propertycount")

        feature = ["Type", "Method", "ParkingArea"]
        for i in feature:
            options = df1[i].dropna().unique().tolist()
            st.selectbox(label=i, options=options, key=i)

    with col2:
        st.subheader("🏙️ Property Size")
        Landsize = st.number_input("Land Size", min_value=0, value=303, key="Landsize")
        BuildingArea = st.number_input("Building Area", min_value=0, value=225, key="BuildingArea")
        # Landsize = st.number_input("Land Size", min_value=1, value=303, key="Landsize")
        # BuildingArea = st.number_input("Building Area", min_value=1, value=225, key="BuildingArea")
        
        st.write("")
        st.subheader("🌍 Location")
        Distance = st.number_input("Distance", min_value=0.0, value=6.4, step=0.1, format="%.3f", key="Distance")
        Latitude = st.number_input("Latitude", min_value=-100.0, value=-37.87, step=0.1, format="%.5f", key="Latitude")
        Longitude = st.number_input("Longitude", min_value=-100.0, value=144.878, step=0.1, format="%.5f", key="Longitude")
        Postcode = st.number_input("Postcode", min_value=0.0, value=3500.0, step=1.0, format="%.1f", key="Postcode")

        feature = ["Regionname"]
        for i in feature:
            options = df1[i].dropna().unique().tolist()
            st.selectbox(label=i, options=options, key=i)

    current_year = datetime.datetime.now().year
    years = list(range(1920, current_year + 1))
    years.reverse()
    YearBuilt = st.selectbox("Select Year", options=years, key="YearBuilt")


with tab2:
    st.text_area("Paste Python Dictionary", height=220, key="dict_input_raw")
    def dictionary_callback():
        raw_text = st.session_state.dict_input_raw
        if raw_text.strip():
            try:
                data = ast.literal_eval(raw_text)
                cast_and_store_state(data)
                st.session_state["tab2_loaded"] = True
            except Exception as e:
                st.session_state["error_msg"] = f"Dictionary Parse Error: {e}"

    st.button("Load Dictionary", on_click=dictionary_callback)
    if "error_msg" in st.session_state:
        st.error(st.session_state.pop("error_msg"))

    if st.session_state.get("tab2_loaded"):
        st.write("---")
        st.markdown("### 🏠 Loaded Property Profile")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Rooms", st.session_state.get("Rooms", "N/A"))
        m_col2.metric("Bathrooms", st.session_state.get("Bathroom", "N/A"))
        m_col3.metric("Car Spaces", st.session_state.get("Car", "N/A"))
        m_col4.metric("Landsize", f"{st.session_state.get('Landsize', 'N/A')} m²")
        
        with st.expander("🔍 View All Loaded Attributes", expanded=False):
            current_input = {k: st.session_state.get(k) for k in ["Rooms", "Type", "Method", "Distance", "Postcode", "Bedroom", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Latitude", "Longitude", "Regionname", "Propertycount", "ParkingArea"]}
            st.dataframe(pd.DataFrame([current_input]), use_container_width=True)
        

with tab3:
    st.text_input("Paste CSV", key="csv_input_raw")
    def csv_callback():
        raw_csv = st.session_state.csv_input_raw
        if raw_csv.strip():
            try:
                values = raw_csv.split(",")
                data = {
                    "Rooms": values[0], "Type": values[1], "Method": values[2],
                    "Distance": values[3], "Postcode": values[4], "Bedroom": values[5],
                    "Bathroom": values[6], "Car": values[7], "Landsize": values[8],
                    "BuildingArea": values[9], "YearBuilt": values[10], "Latitude": values[11],
                    "Longitude": values[12], "Regionname": values[13], "Propertycount": values[14],
                    "ParkingArea": values[15]
                }
                cast_and_store_state(data)
                st.session_state["tab3_loaded"] = True
            except Exception as e:
                st.session_state["csv_error"] = f"CSV Parse Error: {e}"

    st.button("Load CSV", on_click=csv_callback)
    if "csv_error" in st.session_state:
        st.error(st.session_state.pop("csv_error"))

    if st.session_state.get("tab3_loaded"):
        st.write("---")
        st.markdown("### 🏠 Loaded Property Profile")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Rooms", st.session_state.get("Rooms", "N/A"))
        m_col2.metric("Bathrooms", st.session_state.get("Bathroom", "N/A"))
        m_col3.metric("Car Spaces", st.session_state.get("Car", "N/A"))
        m_col4.metric("Landsize", f"{st.session_state.get('Landsize', 'N/A')} m²")
        
        with st.expander("🔍 View All Loaded Attributes", expanded=False):
            current_input = {k: st.session_state.get(k) for k in ["Rooms", "Type", "Method", "Distance", "Postcode", "Bedroom", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Latitude", "Longitude", "Regionname", "Propertycount", "ParkingArea"]}
            st.dataframe(pd.DataFrame([current_input]), use_container_width=True)
                


with tab4:
    def random_house_callback():
        if not df1.empty:
            # Take only rows that have a Price (same as preprocessing)
            valid_df = df1.dropna(subset=["Price"])
            if valid_df.empty:
                st.error("No valid houses found.")
                return

            sample = valid_df.sample(1).iloc[0]
            data = sample.to_dict()

            # Remove Price so it doesn't interfere
            data.pop("Price", None)

            # Handle BuildingArea "inf" or bad values
            if "BuildingArea" in data:
                try:
                    val = float(data["BuildingArea"])
                    if not np.isfinite(val):
                        data["BuildingArea"] = 225   # fallback
                    else:
                        data["BuildingArea"] = val
                except:
                    data["BuildingArea"] = 225

            cast_and_store_state(data)
            st.session_state["tab4_loaded"] = True

    st.button("Load Random", on_click=random_house_callback)

    if st.session_state.get("tab4_loaded"):
        st.write("---")
        st.markdown("### 🏠 Loaded Property Profile")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Rooms", st.session_state.get("Rooms", "N/A"))
        m_col2.metric("Bathrooms", st.session_state.get("Bathroom", "N/A"))
        m_col3.metric("Car Spaces", st.session_state.get("Car", "N/A"))
        m_col4.metric("Landsize", f"{st.session_state.get('Landsize', 0):.1f} m²")
        
        with st.expander("🔍 View All Loaded Attributes", expanded=False):
            current_input = {k: st.session_state.get(k) for k in [
                "Rooms", "Type", "Method", "Distance", "Postcode", "Bedroom", 
                "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", 
                "Latitude", "Longitude", "Regionname", "Propertycount", "ParkingArea"
            ]}
            st.dataframe(pd.DataFrame([current_input]), use_container_width=True)


with tab5:
    # Use only rows that have Price
    valid_len = len(df1.dropna(subset=["Price"])) - 1
    if valid_len < 0:
        valid_len = 0

    st.number_input("Dataset Row", 0, valid_len, 0, key="selected_row_idx")
    
    def load_row_callback():
        if not df1.empty:
            valid_df = df1.dropna(subset=["Price"]).reset_index(drop=True)
            row_idx = st.session_state.selected_row_idx

            if row_idx >= len(valid_df):
                st.error("Row index out of range.")
                return

            sample = valid_df.iloc[row_idx]
            data = sample.to_dict()
            data.pop("Price", None)

            # Handle BuildingArea "inf" or bad values
            if "BuildingArea" in data:
                try:
                    val = float(data["BuildingArea"])
                    if not np.isfinite(val):
                        data["BuildingArea"] = 225
                    else:
                        data["BuildingArea"] = val
                except:
                    data["BuildingArea"] = 225

            cast_and_store_state(data)
            st.session_state["tab5_loaded"] = True
            
    st.button("Load Row", on_click=load_row_callback)

    if st.session_state.get("tab5_loaded"):
        st.write("---")
        st.markdown("### 🏠 Loaded Property Profile")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Rooms", st.session_state.get("Rooms", "N/A"))
        m_col2.metric("Bathrooms", st.session_state.get("Bathroom", "N/A"))
        m_col3.metric("Car Spaces", st.session_state.get("Car", "N/A"))
        m_col4.metric("Landsize", f"{st.session_state.get('Landsize', 0):.1f} m²")
        
        with st.expander("🔍 View All Loaded Attributes", expanded=False):
            current_input = {k: st.session_state.get(k) for k in [
                "Rooms", "Type", "Method", "Distance", "Postcode", "Bedroom", 
                "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", 
                "Latitude", "Longitude", "Regionname", "Propertycount", "ParkingArea"
            ]}
            st.dataframe(pd.DataFrame([current_input]), use_container_width=True)




user_input = {
    key: st.session_state.get(key, default_val)
    for key, default_val in [
        ("Rooms", 4), ("Type", "h"), ("Method", "S"), ("Distance", 6.4),
        ("Postcode", 3500.0), ("Bedroom", 3), ("Bathroom", 2), ("Car", 1),
        ("Landsize", 303), ("BuildingArea", 225), ("YearBuilt", 2000),
        ("Latitude", -37.87), ("Longitude", 144.878), 
        ("Regionname", "Southern Metropolitan"), ("Propertycount", 4030),
        ("ParkingArea", "Detached Garage")
    ]
}


def predict_house_price(user_input: dict):
    input_df = pd.DataFrame([user_input])
    input_df["BuildingArea"] = pd.to_numeric(input_df["BuildingArea"], errors="coerce")

    # Same transforms as training
    for col in ["BuildingArea", "Landsize"]:
        input_df[col] = np.log1p(input_df[col].clip(lower=0))

    # Identical to notebook: drop_first=True
    ohe_cols = ["Type", "Method", "Regionname", "ParkingArea"]
    input_df = pd.get_dummies(input_df, columns=ohe_cols, drop_first=True)

    bool_cols = input_df.select_dtypes(include="bool").columns
    input_df[bool_cols] = input_df[bool_cols].astype(int)
    input_df = input_df.astype(float)

    model, feature_names = load_model()
    # Any missing OHE column (unknown category) → 0 = reference level
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred_log = model.predict(input_df)[0]
    return float(np.expm1(pred_log))



col1, col2, col3 = st.columns([2.5, 3, 1])
with col2: 
    st.markdown('<div class="predict-box-wrapper">', unsafe_allow_html=True)
    # st.info("Note: Model metrics (R², MAE, RMSE) were calculated on log-transformed prices.")
    if st.button("Predict Price", type="primary"):
        predicted_price = predict_house_price(user_input)
        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-title">Estimated Market Value</div>
            <h2 class="prediction-value">{predicted_price:,.2f} AUD</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

