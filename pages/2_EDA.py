import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

BASE_DIR = Path.cwd()
df = pd.read_csv(BASE_DIR / "Data" / "Melbourne_housing.csv", on_bad_lines="skip")

st.divider()
c1, c2 = st.columns(2)
with st.container() :
    with c1 : 
        st.title("📊 Exploratory Data Analysis")
        st.write(
            "##### Explore interactive visualizations and statistical analyses to understand the characteristics of Melbourne's "
            "housing market. This dashboard highlights feature distributions, missing values, correlations, geographic trends, "
            "and relationships with house prices, providing the foundation for building an accurate machine learning model."
        )
    with c2 :
        st.image("https://images.pexels.com/photos/9034997/pexels-photo-9034997.jpeg", width = 320, caption = 'EDA is fun???? 🫠')
        st.write("### Discover Insights from the Melbourne Housing Market Dataset")

st.divider()

def Plot() :
    for i in ax.containers:
        ax.bar_label(i, color = 'red', padding = 3)
    plt.xticks(rotation=60)
    ax.set_title(f"{feature} Distribution", color = 'blue', fontsize=20, fontweight='bold')
    ax.set_xlabel(f"{feature}", color = 'red', fontsize=16, fontweight='bold')
    ax.set_ylabel("Count", color = 'red', fontsize=16, fontweight='bold')
    st.pyplot(fig)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Dataset",
        "📈 Distribution",
        "📊 Relationships",
        "🌍 Geography"
    ]
)


with tab1 :
    st.subheader("📋 Dataset Preview")
    st.write("###### Preview the first five rows of the raw Melbourne Housing dataset.")
    st.dataframe(df.head(), use_container_width=True)
    with st.expander("View Complete Dataset"):
        st.dataframe(df)

    st.subheader("📊 Raw Dataset Summary")
    rows = df.shape[0]
    columns = df.shape[1]
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", columns)
    col3.metric("Missing Values", missing)
    col4.metric("Duplicate Rows", duplicates)
    st.subheader("📑 Statistical Summary")
    st.dataframe(df.describe())
    st.divider()


    st.subheader("🧩 Missing Values Analysis")
    fig, ax = plt.subplots(figsize=(15, 13))
    missing = df.isnull().sum()
    missing[missing > 0].sort_values().plot(kind="bar", color = 'purple', ax=ax)
    for i in ax.containers:
        ax.bar_label(i, color='red')
    
    ax.set_title('Missing Values Analysis', fontsize=20, fontweight='bold', color='red')
    ax.set_xlabel('Features', color='blue', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', color='blue', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    # st.bar_chart(missing)


with tab2 : 
    st.subheader("💰 Price Distribution")
    bins = st.slider(
        "Number of bins",
        min_value=10,
        max_value=50,
        value=30,
        key="slider_first"
    )

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.histplot(
        df["Price"],
        kde=True,
        bins=bins,
        color = 'Green',
        ax=ax
    )

    for i in ax.containers:
        ax.bar_label(i, color = 'red', rotation=90, padding=3)
    ax.set_title("Distribution of House Prices", color = 'blue', fontsize=20, fontweight='bold')
    ax.set_xlabel("Price", color = 'red', fontsize=16, fontweight='bold')
    ax.set_ylabel("Count", color = 'red', fontsize=16, fontweight='bold')
    st.pyplot(fig)


    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(15,13))
    dummy1 = df[['Rooms', 'Distance', 'Postcode', 'Bedroom', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'Latitude', 'Longitude', 'Propertycount', 'Price']]
    sns.heatmap(dummy1.corr(), cmap="plasma", annot=True, fmt = ".4f",  ax=ax)
    plt.title('Correlation between Numeric features with Price columns', color = 'red', fontsize = 16, fontweight = 'bold')
    st.pyplot(fig)

    st.subheader("📈 Numerical Feature Distribution")
    feature = st.selectbox(
        "Select Numerical Feature",
        [
            "Rooms",
            "Bedroom",
            "Bathroom",
            "Car"
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.countplot(
        data=df,
        x=feature,
        color="maroon",
        ax=ax
    )
    Plot()
    
    feature = st.selectbox(
    "Select Numerical Feature for HistPlot",
    [
        "Distance",
        "Propertycount"
    ])

    bins = st.slider(
        "Number of bins",
        min_value=10,
        max_value=50,
        value=30,
        key="slider_second"
    )

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.histplot(
        data = df,
        x=feature,
        # kde=True,
        bins=bins,
        color = 'Green',
        ax=ax
    )

    for i in ax.containers:
        ax.bar_label(i, padding=4, color = 'red', rotation = 90)
    plt.xticks(rotation=60)
    ax.set_title(f"{feature} Distribution", color = 'blue', fontsize=20, fontweight='bold')
    ax.set_xlabel(f"{feature}", color = 'red', fontsize=16, fontweight='bold')
    ax.set_ylabel("Count", color = 'red', fontsize=16, fontweight='bold')
    st.pyplot(fig)


    st.subheader("📊 Categorical Feature Distribution")
    feature = st.selectbox(
        "Select Feature",
        [
            "Type",
            "Method",
            "Regionname",
            "CouncilArea"
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.countplot(
        data=df,
        x=feature,
        color="purple",
        ax=ax
    )
    Plot()


with tab3:
    st.subheader("📈 Feature Relationship with Price")
    feature = st.selectbox("Select Feature",["Rooms", "Bedroom", "Bathroom", "Postcode", "Car", "YearBuilt", "BuildingArea", "Distance", "Landsize", "Latitude", "Longitude","Propertycount"])

    df1 = df.copy()
    df1['BuildingArea'] = pd.to_numeric(df1['BuildingArea'], errors='coerce')

    fig, ax = plt.subplots(figsize=(11,9))
    sns.scatterplot(
        data=df1,
        x=feature,
        y=df1["Price"],
        marker='o',
        color = 'brown',
        alpha=0.6,
        ax=ax)

    sns.regplot(
        data=df1,
        x=feature,
        y=df1["Price"],
        scatter=False,
        color="red",
        ax=ax)

    plt.title(f'{feature} vs Price', color = 'red', fontsize = 20, fontweight = 'bold')
    plt.xlabel(f'{feature}', color = 'b', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Price', color = 'b', fontsize = 16, fontweight = 'bold')
    plt.show()
    plt.tight_layout()
    st.pyplot(fig)

    # df_clean = df[['Rooms', 'Bathroom', 'Bedroom', 'Distance', 'Car', 'Price']].dropna()
    # fig = sns.pairplot(data=df_clean)
    # st.pyplot(fig)

    st.subheader("📦 House Price by Categorical Features")
    feature = st.selectbox(
        "Select a Categorical Feature",
        [
            "Type",
            "Method",
            "Regionname",
            "CouncilArea"
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.boxplot(
        data=df,
        x="Price",
        y=feature,
        palette="Set2",
        ax=ax
    )

    ax.set_title(
        f"House Price by {feature}",
        color ='red',
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel("House Price", color = 'blue', fontsize = 16, fontweight = 'bold')
    ax.set_ylabel(feature, color = 'blue', fontsize = 16, fontweight = 'bold')

    # Cheat Sheet
    cheat_sheet = (
        "📖 How to Read a Box Plot\n\n"
        "• Middle Line = Median Price\n"
        "• Box = Middle 50% of Properties\n"
        "• Whiskers = Normal Price Range\n"
        "• Dots = Outliers"
    )

    ax.text(
        0.75,
        0.02,
        cheat_sheet,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            facecolor="lightyellow",
            alpha=0.9
        )
    )
    # plt.tight_layout()
    st.pyplot(fig)



with tab4:
    st.subheader("🌍 Geographic Distribution")
    fig, ax = plt.subplots(figsize=(15, 13))
    scatter = ax.scatter( 
        x=df["Longitude"],
        y=df["Latitude"],
        c=df["Price"],
        s=df["Rooms"] * 18,
        cmap="plasma",
        alpha=0.65,
        edgecolors="white",
        linewidth=0.5
    )

    plt.colorbar(scatter, label="Price", ax=ax)
    ax.set_title("Melbourne Housing Prices by Geographic Location", color = 'blue',  fontsize=20, fontweight='bold')
    ax.set_xlabel("Longitude", color = 'red', fontsize=14, fontweight='bold')
    ax.set_ylabel("Latitude", color = 'red', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    st.subheader("🏆 Top 10 Expensive Suburbs")
    top = (
        df.groupby("Suburb")["Price"]
        .mean()
        .sort_values(ascending=False)
        .head(10))

    st.write(top)
    st.markdown("### Bar Graph")
    fig, ax = plt.subplots(figsize=(16,10))
    sns.barplot(x=top.values, y=top.index, palette="Reds_r", ax=ax)
    st.pyplot(fig)

    st.subheader("💲 Top 10 Affordable Suburbs")
    cheap = (
        df.groupby("Suburb")["Price"]
        .mean()
        .sort_values(ascending=True)
        .head(10))
    
    st.write(cheap)
    st.markdown("### Bar Graph")
    fig, ax = plt.subplots(figsize=(16,10))
    sns.barplot(x=cheap.values, y=cheap.index, palette=['violet'], ax=ax)
    st.pyplot(fig)