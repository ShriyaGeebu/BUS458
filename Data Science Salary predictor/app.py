#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# In[95]:


def parse_salary(s) -> float:
    """Convert salary bracket strings to numeric averages."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s or '?' in s:
        return None
    if s.startswith('$'):
        s = s[1:]
    if s.startswith('>'):
        val = s[1:].lstrip('$').replace(',', '')
        try:
            return float(val)
        except:
            return None
    if s.startswith('<'):
        val = s[1:].lstrip('$').replace(',', '')
        try:
            return float(val) / 2
        except:
            return None
    if '-' in s:
        try:
            low, high = s.split('-')
            low_f = float(low.lstrip('$').replace(',', ''))
            high_f = float(high.lstrip('$').replace(',', ''))
            return (low_f + high_f) / 2
        except:
            return None
    return None

def load_data(path: str) -> pd.DataFrame:
    """Load survey data, clean columns, and parse salary to numeric."""
    df = pd.read_csv(path, low_memory=False)
    df = df[['Q2', 'Q4', 'Q23', 'Q29']].copy()
    df.columns = ['AgeGroup', 'Country', 'ExperienceLevel', 'SalaryRaw']
    # Remove header/prompt rows and NaNs
    df = df[df['AgeGroup'] != 'What is your age (# years)?']
    df = df[~df['ExperienceLevel'].str.contains(r'Select the title', na=False)]
    df = df[df['SalaryRaw'].notna()]
    df = df[~df['SalaryRaw'].str.contains(r'\?', na=False)]
    # Parse salary to numeric
    df['Salary'] = df['SalaryRaw'].apply(parse_salary)
    return df.dropna(subset=['Salary'])[['AgeGroup', 'Country', 'ExperienceLevel', 'Salary']]

def build_model(df: pd.DataFrame):
    """Train a Random Forest on categorical inputs and return pipeline + R²."""
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
    return pipeline, pipeline.score(X_test, y_test)

def predict_salary(model, age_group: str, country: str, exp: str) -> float:
    """Given trained model and params, return predicted salary."""
    df_input = pd.DataFrame({
        'AgeGroup': [age_group],
        'Country': [country],
        'ExperienceLevel': [exp]
    })
    return model.predict(df_input)[0]


# In[97]:


file_path = 'kaggle_survey_2022_responses.csv'

# 1) Load & clean:
df = load_data(file_path)

# 2) Train & evaluate:
model, score = build_model(df)
print(f"Model R² score: {score:.2f}")

# 3) Example prediction:
ag = df['AgeGroup'].mode()[0]
co = df['Country'].mode()[0]
ex = df['ExperienceLevel'].mode()[0]
pred = predict_salary(model, ag, co, ex)
print(f"Example prediction for {ag}, {co}, {ex}: ${pred:,.0f}")


# In[ ]:




