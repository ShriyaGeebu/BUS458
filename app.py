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
        return float(s[1:].replace(',', ''))
    if s.startswith('<'):
        return float(s[1:].replace(',', ''))/2
    if '-' in s:
        low, high = s.split('-')
        return (float(low.replace(',', '')) + float(high.replace(',', '')))/2
    return None

@st.cache_data
def load_data(path: str = 'kaggle_survey_2022_responses.csv') -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[['Q2','Q4','Q23','Q29']].copy()
    df.columns = ['AgeGroup','Country','ExperienceLevel','SalaryRaw']
    df = df[df['AgeGroup'] != 'What is your age (# years)?']
    df = df[~df['ExperienceLevel'].str.contains(r'Select the title', na=False)]
    df = df[df['SalaryRaw'].notna()]
    df = df[~df['SalaryRaw'].str.contains(r'\?', na=False)]
    # ← now parse_salary is defined above, so this will work:
    df['Salary'] = df['SalaryRaw'].apply(parse_salary)
    return df.dropna(subset=['Salary'])[['AgeGroup','Country','ExperienceLevel','Salary']]

@st.cache_resource
def build_model(df):
    # …same as before…
    …

def predict_salary(model, age, country, exp):
    # …same as before…
    …

# ——— Your Streamlit UI below ———
st.title("🎯 Data Science Salary Predictor")
df = load_data()
model = build_model(df)
# …