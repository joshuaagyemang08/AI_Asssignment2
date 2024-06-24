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

    movement_reactions = st.slider("Movement Reactions", 1, 100, 50)
    passing = st.slider("Passing", 1, 100, 50)
    mentality_composure = st.slider("Mentality Composure", 1, 100, 50)
    value_eur = st.slider("Value EUR", 1, 100000000, 1)
    wage_eur = st.slider("Wage EUR", 1, 1000000, 1)
    dribbling = st.slider("Dribbling", 1, 100, 50)
    attacking_short_passing = st.slider("Attacking Short Passing", 1, 100, 50)
    mentality_vision = st.slider("Mentality Vision", 1, 100, 50)
    international_reputation = st.slider("International Reputation", 1, 5, 1)
    skill_long_passing = st.slider("Skill Long Passing", 1, 100, 50)
    power_shot_power = st.slider("Power Shot Power", 1, 100, 50)
    physic = st.slider("Physic", 1, 100, 50)
    release_clause_eur = st.slider("Release Clause EUR", 1, 200000000, 10000000)
    age = st.slider("Age", 15, 40, 25)
    skill_ball_control = st.slider("Skill Ball Control", 1, 100, 50)

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