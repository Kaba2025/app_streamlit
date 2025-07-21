

##____________________SUCCES____________________________________________________________-

import streamlit as st
import pandas as pd
import joblib

# ======================
# 1. CONFIGURATION PAGE
# ======================
st.set_page_config(page_title="🎓 Orientation Post-Bac", layout="wide")
st.markdown("""
    <style>
    body {background-color: #F8F9FA;}
    .stButton button {background-color: #004AAD; color: white; font-size: 18px; border-radius: 10px; width: 100%;}
    .result-card {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .badge-success {
        background-color: #28A745;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        display: inline-block;
        width: 100%;
    }
    .badge-danger {
        background-color: #DC3545;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        display: inline-block;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# 2. CHARGER LE MODÈLE
# ======================
model = joblib.load("random_forest_orientation.pkl")  # Mets ton chemin ici
data_ref = pd.read_excel("ORIEN.xlsx")  # Mets ton chemin ici

# ======================
# 3. HEADER
# ======================
st.title("🔍 Analyse & Prédiction d’Orientation Post-Bac")
st.write("💡 Entrez les informations de l’étudiant pour savoir si le choix est adapté.")

# ======================
# 4. LAYOUT À DEUX COLONNES
# ======================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Informations Étudiant")

    
    ville = st.text_input("📍 Ville du Bac")
    filiere = st.text_input("🎓 Quelle filière souhaitez-vous faire ?")
    motif_filiere = st.selectbox("🤔 Pourquoi ce choix ?", options=sorted(data_ref['motif_filière'].unique()))

    preparation_choix = st.radio("🧠 Avez-vous préparé ce choix ?", ["Oui", "Non", "Un peu"], horizontal=True)
    connaissance_filiere = st.radio("📚 Connaissance de la filière ?", ["Oui", "Non", "Un peu"], horizontal=True)
    conseil_par_pro = st.radio("👔 Conseillé par un professionnel ?", ["Oui", "Non"], horizontal=True)
    acces_plateforme = st.radio("🌐 Avez-vous Utilisé une plateforme ?", ["Oui", "Non"], horizontal=True)
    satisfaction_filiere = st.radio("😊 Pensez vous que vous serez Satisfait(e) de la filière ?", ["Oui", "Non"], horizontal=True)
    diplome = st.radio("🎓 Avez-vous un diplôme ?", ["Oui", "Non"], horizontal=True)
    employabilite = st.radio("💼 Pensez-vous que vous serez Employable ?", ["Oui", "Non"], horizontal=True)
    reconversion = st.radio("🔄 Pensez-vous a une reconversion dans le futur?", ["Oui", "Non"], horizontal=True)
    referiez_meme_choix = st.radio("🔁 Referiez-vous ce choix ?", ["Oui", "Non"], horizontal=True)

    # Bouton de prédiction
    predict_btn = st.button("🎯 Prédire le choix")

with col2:
    st.subheader("📊 Résultats")
    placeholder = st.empty()

# ======================
# 5. PRÉDICTION
# ======================
if predict_btn:
    # Préparer les données pour le modèle
    input_data = pd.DataFrame({
        'preparation_choix': [preparation_choix],
        'Connaissance_filière': [connaissance_filiere],
        'conseil_par_pro': [conseil_par_pro],
        'accès_plateforme': [acces_plateforme],
        'satisfaction_filière': [satisfaction_filiere],
        'diplôme': [diplome],
        'employabilite': [employabilite],
        'reconversion': [reconversion],
        'referiez_même_choix': [referiez_meme_choix]
    })

    conversion = {"Oui": 1, "Non": 0, "Un peu": 0.5}
    input_data = input_data.applymap(lambda x: conversion.get(x, x))

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    score_bon = proba[1] * 100
    score_mauvais = proba[0] * 100

    # ======================
    # 6. AFFICHAGE RÉSULTAT
    # ======================
    with placeholder.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<p class="badge-success">✅ Bon choix détecté</p>', unsafe_allow_html=True)
            st.progress(int(score_bon))
            st.write(f"Score : **{score_bon:.2f}%**")
            commentaire = "👍 Continuez dans cette voie ! Vous avez bien préparé votre choix."
            recommandations = "- Continuez à vous informer sur la filière.\n- Maintenez votre motivation."
        else:
            st.markdown('<p class="badge-danger">❌ Risque de mauvais choix</p>', unsafe_allow_html=True)
            st.progress(int(score_mauvais))
            st.write(f"Score : **{score_mauvais:.2f}%**")
            commentaire = "⚠️ Votre choix semble peu préparé. Réfléchissez à d’autres options."
            recommandations = "- Consultez un conseiller d’orientation.\n- Explorez les débouchés du secteur."

        st.write("### 📝 Commentaire :")
        st.info(commentaire)

        st.write("### ✅ Recommandations :")
        st.success(recommandations)

        st.markdown('</div>', unsafe_allow_html=True)


























