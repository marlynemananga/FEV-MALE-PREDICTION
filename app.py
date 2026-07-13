import streamlit as st
import joblib
import pandas as pd

# Chargement du modèle enregistré
model = joblib.load("mlp_model.pkl")

# Titre de l'application
st.title("Prédiction du FEV par un réseau de neurones")

st.write("Entrez les caractéristiques du patient")

# Entrées utilisateur
age = st.number_input(
    "Âge (années)",
    min_value=4,
    max_value=87,
    value=40
)

height = st.number_input(
    "Taille (cm)",
    min_value=103,
    max_value=196,
    value=160
)

# Bouton de prédiction
if st.button("Prédire le FEV"):

    donnees = pd.DataFrame(
        [[age, height]],
        columns=["age", "height"]
    )

    prediction = model.predict(donnees)

    st.success(
        f"FEV prédit : {prediction[0]:.3f}"
    )