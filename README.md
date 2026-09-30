# 🏠 Melbourne House Price Prediction

> An end-to-end Machine Learning project that predicts residential property prices in **Melbourne, Australia** using **_Linear Regression_**.

The project includes **exploratory data analysis, data preprocessing, model training & evaluation**, and an interactive **Streamlit web application** for price prediction.

---

## ✨ Features

- 📊 **Interactive Exploratory Data Analysis**
  - Distributions
  - Correlations
  - Relationships
  - Geographic insights

- 🧹 **Complete Data Cleaning & Preprocessing Pipeline**

- 🤖 **Linear Regression Model**
  - Model training
  - Performance evaluation
  - Evaluation metrics

- 🔮 **Interactive Price Prediction**
  - Manual entry
  - Paste Python dictionary
  - Paste CSV row
  - Load random house
  - Load specific dataset row

- 🖥️ **Clean Multipage Streamlit Interface**
  - Home
  - EDA
  - Prediction

---

## 📁 Project Structure

```text
melbourne-house-price-prediction/
│
├── App.py                         # Main Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── Data/
│   ├── Melbourne_housing.csv      # Raw dataset
│   └── Cleaned_dataset.csv        # Preprocessed dataset
│
├── Model/
│   ├── feature_names.pkl          # Feature names used by the model
│   ├── linear_regression.pkl      # Trained Linear Regression model
│   ├── linear_regression.joblib   # (optional duplicate)
│   └── metrics.json               # Model evaluation metrics
│
├── Notebook/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model Run.ipynb
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_EDA.py
│   └── 3_Prediction.py
│
└── Structure/
    └── module.txt
```

---

## 📓 Notebooks

The project is organized into three main notebooks:

| Notebook                 | Purpose                         |
| ------------------------ | ------------------------------- |
| `01_EDA.ipynb`           | Exploratory Data Analysis       |
| `02_Preprocessing.ipynb` | Data cleaning and preprocessing |
| `03_Model Run.ipynb`     | Model training and evaluation   |

---

## 📌 Notes

> **Important:** All evaluation metrics are on **log-transformed prices**.

Prediction applies the **same preprocessing as the training notebooks** to ensure consistency between training and real-world predictions.

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd melbourne-house-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run App.py
```

---

## 🛠️ Technologies Used

| Category               | Technology                       |
| ---------------------- | -------------------------------- |
| 🐍 Language            | Python                           |
| 📦 Data Manipulation   | NumPy, Pandas                    |
| 📊 Visualization       | Matplotlib, Seaborn              |
| 🤖 Machine Learning    | Scikit-learn (Linear Regression) |
| 💾 Model Serialization | Joblib                           |
| 🌐 Web App             | Streamlit                        |
| 📓 Environment         | Jupyter Notebook, VS Code        |
| 🔧 Version Control     | Git & GitHub                     |

---

## 🔮 How Prediction Works

The prediction pipeline follows the same preprocessing logic used during model training.

```text
                 ┌─────────────────┐
                 │    User Input   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Validation   │
                 └────────┬────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Same preprocessing used │
              │    in the notebooks     │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │ One-Hot Encoding│
                 └────────┬────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Column Alignment with  │
              │   training features    │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │ Model Prediction│
                 │    (log scale)  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Inverse Log     │
                 │ Transformation  │
                 │    (expm1)      │
                 └────────┬────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Predicted Price      │
              │         (AUD)           │
              └─────────────────────────┘
```

---

## 📈 Prediction Pipeline

```text
User Input
    ↓
Validation
    ↓
Same preprocessing used in notebooks
    ↓
One-Hot Encoding
    ↓
Column Alignment
    ↓
Linear Regression Model
    ↓
Log-scale Prediction
    ↓
Inverse Log Transformation (expm1)
    ↓
Predicted Price (AUD)
```

---

## 👨‍💻 Author

**Mohi Uddin Maruf**

- 🔗 **GitHub:** https://github.com/Maruf39237
- 📧 **Email:** [maruf39237@gamil.com](mailto:maruf39237@gamil.com)

---
