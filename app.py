import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def parse_salary(s: str) -> float:
    """Convert salary bracket strings to numeric averages."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s or '?' in s:
        return None
    if s.startswith('$'):
        s = s[1:]
    if s.startswith('>'):
        # e.g. ">200000"
        try:
            return float(s[1:].replace(',', ''))
        except ValueError:
            return None
    if s.startswith('<'):
        # e.g. "<50000" -> average ~25000
        try:
            return float(s[1:].replace(',', '')) / 2
        except ValueError:
            return None
    if '-' in s:
        # e.g. "50000-59999"
        try:
            low, high = s.split('-')
            low_f = float(low.replace(',', ''))
            high_f = float(high.replace(',', ''))
            return (low_f + high_f) / 2
        except ValueError:
            return None
    return None

@st.cache_data
def load_data(path: str = 'kaggle_survey_2022_responses.csv') -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[['Q2', 'Q4', 'Q23', 'Q29']].copy()
    df.columns = ['AgeGroup', 'Country', 'ExperienceLevel', 'SalaryRaw']
    # Filter out header prompts and missing
    df = df[df['AgeGroup'] != 'What is your age (# years)?']
    df = df[~df['ExperienceLevel'].str.contains(r'Select the title', na=False)]
    df = df[df['SalaryRaw'].notna()]
    df = df[~df['SalaryRaw'].str.contains(r'\?', na=False)]
    # Parse numeric salary
    df['Salary'] = df['SalaryRaw'].apply(parse_salary)
    return df.dropna(subset=['Salary'])[['AgeGroup', 'Country', 'ExperienceLevel', 'Salary']]

@st.cache_resource
def build_model(df: pd.DataFrame):
    X = df[['AgeGroup', 'Country', 'ExperienceLevel']]
    y = df['Salary']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
         ['AgeGroup', 'Country', 'ExperienceLevel'])
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_salary(model, age_group: str, country: str, exp_level: str) -> float:
    df_input = pd.DataFrame({
        'AgeGroup': [age_group],
        'Country': [country],
        'ExperienceLevel': [exp_level]
    })
    return model.predict(df_input)[0]

# ─── Streamlit UI ───────────────────────────────────
st.title("🎯 Data Science Salary Predictor")

# 1) Load data and model once
df = load_data()
model = build_model(df)

# 2) Sidebar inputs
st.sidebar.header("Enter your profile")
age = st.sidebar.selectbox("Age group", df['AgeGroup'].unique())
country = st.sidebar.selectbox("Country", df['Country'].unique())
exp = st.sidebar.selectbox("Experience level", df['ExperienceLevel'].unique())

# 3) Predict on button click
if st.sidebar.button("Predict salary"):
    salary = predict_salary(model, age, country, exp)
    st.metric("💰 Predicted Annual Salary", f"${salary:,.0f}")
else:
    st.write("Select your profile on the left and click **Predict salary**.")