import streamlit as st

st.set_page_config(
    page_title="Melbourne House Price Prediction",
    page_icon="🏠",
    # layout="wide",
    # initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        [data-testid="stSidebarNavItems"] span {
            font-size: 18px !important;
            font-weight: bold;
        }
        [data-testid="stSidebar"] {
            width: 300px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

home = st.Page("pages/1_Home.py", title="Home", icon="🏠")
eda  = st.Page("pages/2_EDA.py",  title="Exploratory Data Analysis", icon="📊")
pred = st.Page("pages/3_Prediction.py", title="Prediction", icon="🔮")

pg = st.navigation([home, eda, pred], position="sidebar")
pg.run()