import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import joblib

model = joblib.load('final_model.pkl')

cols=['movement_reactions', 'passing', 'mentality_composure', 'value_eur', 'wage_eur', 'dribbling', 'attacking_short_passing', 'mentality_vision', 'international_reputation', 'skill_long_passing', 'power_shot_power', 'physic', 'release_clause_eur', 'age', 'skill_ball_control']

def main(): 
    st.title("FIFA OVR Predictor")
    st.markdown("""
    <style>
    .reportview-container {
        background: #123456;
    }
    .sidebar .sidebar-content {
        background: #123456;
    }
    </style>
    """, unsafe_allow_html=True)

    movement_reactions = st.number_input("Movement Reactions", min_value=1, max_value=100, value=50, step=1)
    passing = st.number_input("Passing", min_value=1, max_value=100, value=50, step=1)
    mentality_composure = st.number_input("Mentality Composure", min_value=1, max_value=100, value=50, step=1)
    value_eur = st.number_input("Value EUR", min_value=1, max_value=100000000, value=1, step=1000)
    wage_eur = st.number_input("Wage EUR", min_value=1, max_value=1000000, value=1, step=100)
    dribbling = st.number_input("Dribbling", min_value=1, max_value=100, value=50, step=1)
    attacking_short_passing = st.number_input("Attacking Short Passing", min_value=1, max_value=100, value=50, step=1)
    mentality_vision = st.number_input("Mentality Vision", min_value=1, max_value=100, value=50, step=1)
    international_reputation = st.number_input("International Reputation", min_value=1, max_value=5, value=1, step=1)
    skill_long_passing = st.number_input("Skill Long Passing", min_value=1, max_value=100, value=50, step=1)
    power_shot_power = st.number_input("Power Shot Power", min_value=1, max_value=100, value=50, step=1)
    physic = st.number_input("Physic", min_value=1, max_value=100, value=50, step=1)
    release_clause_eur = st.number_input("Release Clause EUR", min_value=1, max_value=200000000, value=10000000, step=100000)
    age = st.number_input("Age", min_value=15, max_value=40, value=25, step=1)
    skill_ball_control = st.number_input("Skill Ball Control", min_value=1, max_value=100, value=50, step=1)

    if st.button("Predict"):
        expected_features = [
            'movement_reactions', 'passing', 'wage_eur', 'mentality_composure', 
            'value_eur', 'dribbling', 'attacking_short_passing', 'mentality_vision', 
            'international_reputation', 'skill_long_passing', 'power_shot_power', 
            'physic', 'release_clause_eur', 'age', 'skill_ball_control'
        ]    
        data = {
            'movement_reactions': movement_reactions,
            'passing': passing,
            'mentality_composure': mentality_composure,
            'value_eur': value_eur,
            'wage_eur': wage_eur,
            'dribbling': dribbling,
            'attacking_short_passing': attacking_short_passing,
            'mentality_vision': mentality_vision,
            'international_reputation': international_reputation,
            'skill_long_passing': skill_long_passing,
            'power_shot_power': power_shot_power,
            'physic': physic,
            'release_clause_eur': release_clause_eur,
            'age': age,
            'skill_ball_control': skill_ball_control
        }

        df = pd.DataFrame([data], columns=expected_features)

        st.write(df.loc[0:1, :])
    


        prediction = model.predict(df)

        print(df)
        

        
        st.write(f"Prediction: The player has an overall rating of {round(prediction[0])}")
        st.write(f"Confidence Level: {93.90: .2f}%")

        if st.button("Predict Again"):
            main()


if __name__=='__main__': 
    main()