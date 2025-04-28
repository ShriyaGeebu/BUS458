import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ————————————————————————————————
# 1) Helper functions (as you already have them)
# ————————————————————————————————
@st.cache_data
def load_data(path: str = 'kaggle_survey_2022_responses.csv') -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[['Q2','Q4','Q23','Q29']].copy()
    df.columns = ['AgeGroup','Country','ExperienceLevel','SalaryRaw']
    df = df[df['AgeGroup'] != 'What is your age (# years)?']
    df = df[~df['ExperienceLevel'].str.contains(r'Select the title', na=False)]
    df = df[df['SalaryRaw'].notna()]
    df = df[~df['SalaryRaw'].str.contains(r'\?', na=False)]
    df['Salary'] = df['SalaryRaw'].apply(parse_salary)
    return df.dropna(subset=['Salary'])[['AgeGroup','Country','ExperienceLevel','Salary']]

@st.cache_resource
def build_model(df: pd.DataFrame):
    X = df[['AgeGroup','Country','ExperienceLevel']]
    y = df['Salary']
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.columns.tolist())
    ])
    pipe = Pipeline([
        ('pre', pre),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)
    pipe.fit(X_tr, y_tr)
    return pipe

def predict_salary(model, age, country, exp):
    df_in = pd.DataFrame({
        'AgeGroup':[age],
        'Country':[country],
        'ExperienceLevel':[exp]
    })
    return model.predict(df_in)[0]

# ————————————————————————————————
# 2) Streamlit UI
# ————————————————————————————————
st.title("🎯 Data Science Salary Predictor")

# Load once
df = load_data()
model = build_model(df)

st.sidebar.header("Your profile")
age = st.sidebar.selectbox("Age group", df['AgeGroup'].unique())
country = st.sidebar.selectbox("Country", df['Country'].unique())
exp = st.sidebar.selectbox("Experience level", df['ExperienceLevel'].unique())

if st.sidebar.button("Predict salary"):
    salary = predict_salary(model, age, country, exp)
    st.metric("💰 Predicted annual salary", f"${salary:,.0f}")


