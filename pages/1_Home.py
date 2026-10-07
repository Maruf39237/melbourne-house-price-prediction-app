import streamlit as st


st.divider()
col1, col2 = st.columns(2)
with st.container():
    with col1 : 
        st.title("Melbourne Housing Price Analytics & Prediction -- 🏠")
        st.header("Interactive Machine Learning Dashboard")
        st.subheader("• Data Exploration")
        st.subheader("• Model Evaluation")
        st.subheader("• Price Prediction")
    with col2  : 
        url1 = "https://ch.mathworks.com/matlabcentral/mlc-downloads/downloads/3d7d93cf-baaf-4c56-b6ae-087fbf15e17f/5126b52f-f6f5-4a1b-8fbf-60703b5cfa93/images/screenshot.jpg"
        st.image(url1, width=320)
        st.subheader(
            "A Streamlit application demonstrating an end-to-end machine learning workflow,"
            "including exploratory data analysis, data preprocessing, model evaluation, "
            "and house price prediction using Linear Regression.")

st.divider()

# ----------------------------------------------------------------------------------------
st.markdown("""### 📖 Project Description
###### --- The Melbourne House Price Prediction --- project is an end-to-end Machine Learning application that predicts residential property prices using the Melbourne Housing dataset. The project follows a complete data science workflow, including exploratory data analysis (EDA), data preprocessing, feature engineering, model development, performance evaluation, and deployment through an interactive Streamlit web application.\n

###### The application allows users to explore housing market trends, visualize important insights, evaluate the Linear Regression model, and predict house prices based on property characteristics such as location, property type, number of rooms, bathrooms, land size, building area, and distance from Melbourne's Central Business District (CBD).""")

st.markdown("""### 🔄 ML Workflow
``` text
                Melbourne Housing Dataset
                            │
                            ▼
                Exploratory Data Analysis (EDA)
                            │
                            ▼
                Data Cleaning & Preprocessing
                    • Missing Value Handling
                    • Duplicate Removal
                    • Outlier Treatment (IQR Capping)
                    • Feature Selection
                    • One-Hot Encoding
                    • Log Transformation
                    • Feature Scaling
                            │
                            ▼
                Train-Test Split (80% / 20%)
                            │
                            ▼
                Linear Regression Model Training
                            │
                            ▼
                Model Performance Evaluation
                    • R² Score
                    • Adjusted R²
                    • MAE
                    • MSE
                    • RMSE
                    • Residual Analysis
                    Note: All evaluation metrics R² score, MAE, MSE, 
                    RMSE etc. are calculated on log-transformed prices.
                            │
                            ▼
                Streamlit Web Application Deployment
```""")

# st.html("""
#         <div style="
#             border-left: 2px solid rgba(49, 51, 63, 0.2); 
#             height: 1550px; 
#             margin: 0 auto;
#             width: 1px;
#         "></div>
#     """)

st.markdown("""### 📊 Dataset Overview

###### The project uses the **Melbourne Housing Dataset**, which contains detailed information about residential properties sold across Melbourne, Australia. The dataset includes numerical and categorical features describing each property, making it suitable for regression analysis and house price prediction.""")


st.markdown("#### Key Features")
c1, c2 = st.columns(2)
with c1 : 
    st.markdown("""
* **Rooms** – Total number of rooms
* **Bedroom** – Number of bedrooms
* **Bathroom** – Number of bathrooms
* **Car** – Number of car parking spaces
* **Landsize** – Land area of the property (m²)
* **BuildingArea** – Building area (m²)
* **YearBuilt** – Construction year
    """)

with c2 :
    st.markdown("""
* **Distance** – Distance from Melbourne CBD (km)
* **Latitude & Longitude** – Geographic location
* **Regionname** – Region of the property
* **Type** – Property type (House, Unit, Townhouse)
* **Method** – Selling method
* **Propertycount** – Number of properties in the suburb
    """)

st.markdown("""
#### Target Variable
* **Price** – Selling price of the property (AUD)
""")

# st.divider()
st.markdown("### 📈 Data Pipeline Dashboard")
hdr_1, hdr_2, hdr_3, hdr_4 = st.columns([2, 2, 2, 3])
hdr_1.markdown("#### Stage")
hdr_2.markdown("#### Rows")
hdr_3.markdown("#### Columns")
hdr_4.markdown("#### Details")
st.divider()

row1_1, row1_2, row1_3, row1_4 = st.columns([2, 2, 2, 3])
row1_1.markdown("### 📁 Raw")
row1_2.metric(label="Total Rows", value="34,857")
row1_3.metric(label="Features", value="22")
row1_4.text("Initial data ingestion") # Placeholder or blank
st.divider()

row2_1, row2_2, row2_3, row2_4 = st.columns([2, 2, 2, 3])
row2_1.markdown("### ✨ Cleaned")
row2_2.metric(label="Final Rows", value="27,239", delta="-7,618")
row2_3.metric(label="Features", value="33", delta="+11") 

with row2_4:
    st.markdown("**Algorithm:** Linear Regression")
    st.markdown("**Deployment:** Streamlit Cloud 🚀")

# st.divider()

st.markdown("""### 🛠 Technologies Used""")
c1, c2 = st.columns(2)
with c1 :
    st.markdown("""
#### Programming Language
* Python

#### Data Manipulation
* NumPy
* Pandas

#### Data Visualization
* Matplotlib
* Seaborn

#### Machine Learning
* Scikit-learn
* Linear Regression""")


with c2 :
    st.markdown("""
#### Model Serialization
* Joblib

#### Web Application
* Streamlit

#### Development Environment
* Jupyter Notebook
* Visual Studio Code

#### Version Control
* Git
* GitHub
""")


