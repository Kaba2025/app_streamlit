

##____________________SUCCES____________________________________________________________-

import streamlit as st
import joblib
import pandas as pd

# 🔐 Charger le meilleur modèle
model = joblib.load(r"C:\INSEEDS\PROJET AFTER BAC\random_forest_orientation.pkl")

# 🖼️ Interface
st.set_page_config(page_title="Orientation Post-Bac", layout="centered")
st.markdown("<h1 style='color: #004aad;'>🔍 Prédiction de bon/mauvais choix de filière</h1>", unsafe_allow_html=True)
st.markdown("<h4>💡 Remplissez les caractéristiques de l'étudiant :</h4>", unsafe_allow_html=True)
kb = pd.read_excel("C:/INSEEDS/ORIEN.xlsx")
# 🎛️ Menus déroulants
villes = sorted(kb['ville_post_bac'].unique())
ville = st.selectbox("Dans quelle ville Avez vous eu votre Bac?", options=villes)
filière= st.text_input("quelle filière aimeriez vous faire? ")
motif_filières = sorted(kb['motif_filière'].unique())
motif_filière = st.selectbox("pourquoi cette filière?", options=motif_filières)
preparation_choix = st.selectbox("🧠 Avez-vous préparé ce choix ?", ["Oui", "Non", "Un peu"])
Connaissance_filière = st.selectbox("📖 connaissez vous bien de la filière", ["Oui", "Non", "Un peu"])
conseil_par_pro = st.selectbox("👔 Avez-vous été conseillé par un professionnel ?", ["Oui", "Non"])
accès_plateforme = st.selectbox("🌐 Avez-vous utilisé une plateforme d’orientation ?", ["Oui", "Non"])
satisfaction_filière = st.selectbox("😊 Seriez-vous satisfait(e) de la filière ?", ["Oui", "Non"])
diplôme = st.selectbox("🎓 Avez-vous un diplôme ?", ["Oui", "Non"])
employabilite = st.selectbox("💼 Pensez-vous que vous serez  ou êtes employable ?", ["Oui", "Non"])
reconversion = st.selectbox("🔄 Souhaitez-vous vous reconvertir ?", ["Oui", "Non"])
referiez_même_choix = st.selectbox("🔁 Referiez-vous le même choix ?", ["Oui", "Non"])


# 🎯 Prédiction au clic
if st.button("🎯 Prédire le choix"):

    # 🧾 Données utilisateur
    input_data = pd.DataFrame({
        'preparation_choix': [preparation_choix],
        'Connaissance_filière': [Connaissance_filière],
        'conseil_par_pro': [conseil_par_pro],
        'accès_plateforme': [accès_plateforme],
        'satisfaction_filière': [satisfaction_filière],
        'diplôme': [diplôme],
        'employabilite': [employabilite],
        'reconversion': [reconversion],
        'referiez_même_choix': [referiez_même_choix]
    })

    # 🔁 Encodage des valeurs
    conversion = {
        "Oui": 1, "Non": 0, "Un peu": 0.5
    }

    input_data = input_data.applymap(lambda val: conversion.get(val, val))  # sécurise les valeurs non mappées

    # 🚨 Vérification
    if input_data.isnull().any().any():
        st.error("⚠️ Certaines réponses ne sont pas reconnues. Vérifie les valeurs ou élargis le mapping.")
        st.write("Données reçues :", input_data)
    else:
        # 🔮 Prédiction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        # 📢 Affichage du résultat
        if prediction == 0:
            st.error(f"❌ Risque élevé : Mauvais choix possible.\n🔻 Score : {proba[0]*100:.2f}%")
        else:
            st.success(f"✅ Bon choix détecté !\n🟢 Score : {proba[1]*100:.2f}%")

        # 📊 Graphique
        st.markdown("### 📊 Score de prédiction :")
        st.bar_chart({"Bon choix": [proba[1]*100], "Mauvais choix": [proba[0]*100]})

