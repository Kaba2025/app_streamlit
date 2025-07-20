import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
# Charger les données
df = pd.read_excel("C:/INSEEDS/ORIEN.xlsx")

# Aperçu rapide
print(df.shape)
print(df.columns)
df.head()

# Liste des variables à examiner
colonnes = [
    'âge', 'Sexe', 'ville_post_bac', 'Profession_pere', 'profession_mere',
    'filière', 'motif_filière', 'choix_filière', 'preparation_choix',
    'Connaissance_filière', 'conseil_par_pro', 'influence_choix',
    'accès_plateforme', 'journées_orientation', 'satisfaction_filière',
    'niveau_étude', 'diplôme', 'employabilite', 'reconversion',
    'referiez_même_choix', 'bon_choix_filière'
]

# Affiche les modalités de chaque variable
for col in colonnes:
    print(f"--- {col} ---")
    print(df[col].unique())
    print()


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Nettoyage manuel des fautes et doublons
df['ville_post_bac'] = df['ville_post_bac'].str.strip().str.title()
df['accès_plateforme'] = df['accès_plateforme'].str.capitalize()

# Harmoniser les accents ou apostrophes mal encodés
df['filière'] = df['filière'].str.strip().str.title().replace({
    "Informatique Développeur D’Applications": "Informatique Développeur D'Applications",
    "L’Anglais": "Anglais"
})
df['motif_filière'] = df['motif_filière'].replace({
    "P Débauchés professionels": "Débauchés professionels"
})
df['journées_orientation'] = df['journées_orientation'].replace({
    "Oui, j'y ai participé": "Oui"
})
df['accès_plateforme'] = df['accès_plateforme'].replace({'oui': 'Oui'})

# 2. Encodage binaire Oui/Non
colonnes_binaires = [
    'preparation_choix', 'Connaissance_filière', 'conseil_par_pro',
    'accès_plateforme', 'satisfaction_filière', 'diplôme', 'employabilite',
    'reconversion', 'referiez_même_choix', 'bon_choix_filière'
]

for col in colonnes_binaires:
    df[col] = df[col].map({'Oui': 1, 'Non': 0, 'Un peu': 0.5})

# 3. Conversion de l'âge en numérique
df['âge'] = pd.to_numeric(df['âge'], errors='coerce')

# 4. Variables catégorielles à encoder
cat_vars = [
    'Sexe', 'ville_post_bac', 'Profession_pere', 'profession_mere',
    'filière', 'motif_filière', 'choix_filière', 'influence_choix',
    'niveau_étude', 'journées_orientation'
]

# Encodage LabelEncoder pour les variables catégorielles
le_dict = {}
for col in cat_vars:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le  # stocker si besoin pour décoder dans l'app

# 5. Vérification finale
print(df.info())
print(df.head())

# Supprimer les variables avec très peu de variation (presque constantes)
from sklearn.feature_selection import VarianceThreshold

# Exclure la cible temporairement
X_temp = df.drop(columns=["bon_choix_filière"])
selector = VarianceThreshold(threshold=0.01)  # < 1% de variance
selector.fit(X_temp)

# Conserver seulement les variables utiles
columns_kept = X_temp.columns[selector.get_support()]
print("Variables avec assez de variance :", list(columns_kept))

import seaborn as sns
import matplotlib.pyplot as plt

# On recolle la cible temporairement
df_corr = df.copy()

# Calcul des corrélations avec la variable cible
correlations = df_corr.corr()['bon_choix_filière'].drop('bon_choix_filière').sort_values(key=abs, ascending=False)

# Affichage des meilleures corrélations
print("Corrélations avec bon_choix_filière :")
print(correlations)

# Visualisation
plt.figure(figsize=(10,6))
sns.barplot(x=correlations.values, y=correlations.index)
plt.title("Corrélation des variables avec bon_choix_filière")
plt.show()

from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["bon_choix_filière"])
y = df["bon_choix_filière"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importance des variables :")
print(importances)

# Graphique
importances.plot(kind='barh', figsize=(10, 8))
plt.title("Importance des variables (Random Forest)")
plt.gca().invert_yaxis()
plt.show()


variables_utiles = [
    'preparation_choix', 'Connaissance_filière', 'conseil_par_pro',
    'accès_plateforme', 'satisfaction_filière', 'diplôme', 'employabilite',
    'reconversion', 'referiez_même_choix'
]

X = df[variables_utiles]
y = df['bon_choix_filière']  # cible


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd

# ⚠️ Étape 1 : rééquilibrage avec SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ⚖️ Étape 2 : split en 3 jeux (train, val, test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
# (→ 52.5% train, 17.5% val, 30% test)

# 🧠 Étape 3 : modèles avec pondération (si applicable)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, penalty='l2', C=0.1, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=5, random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=3, weights='distance')
}

# 🔁 Étape 4 : entraînement + évaluation
results = []
for name, model in models.items():
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1-score': f1,
        'Best Iteration': getattr(model, 'best_iteration_', 'N/A')
    })

# 📊 Étape 5 : affichage des résultats
results_df = pd.DataFrame(results).sort_values('F1-score', ascending=False)
print("RÉSULTATS COMPARATIFS:")
print(results_df.to_string(index=False))

# 🌟 Étape 6 : Importance des variables
print("\nIMPORTANCE DES VARIABLES:")
for name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
    if name in models:
        print(f"\n{name}:")
        importance = pd.Series(models[name].feature_importances_, index=X.columns)
        print(importance.sort_values(ascending=False).head(5))

# 🧾 Étape 7 : évaluation détaillée
print("\nÉVALUATION DÉTAILLÉE PAR MODÈLE :\n")
for name, model in models.items():
    print(f"--- {name} ---")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)
    total = cm.sum()
    correct = cm.diagonal().sum()
    erreurs = total - correct
    taux_erreur = erreurs / total
    print(f"Taux de mauvais classement : {taux_erreur:.2%}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred))
    print("\n")

# 🔎 Trouver le meilleur modèle
best_model_name = results_df.iloc[0]['Model']
print(f"\n🌟 Meilleur modèle selon F1-score : {best_model_name}")

# 🔁 Récupérer l'objet du modèle
best_model = models[best_model_name]

# 🔮 Prédictions
y_best_pred = best_model.predict(X_test)

# 🧾 Matrice de confusion
cm = confusion_matrix(y_test, y_best_pred)
print("Matrice de confusion :")
print(cm)

# ⚠️ Taux de mauvais classement
total = cm.sum()
correct = cm.diagonal().sum()
erreurs = total - correct
taux_erreur = erreurs / total
print(f"Taux de mauvais classement : {taux_erreur:.2%}")

# 📌 Rapport complet
print("\nClassification report :")
print(classification_report(y_test, y_best_pred))
import joblib

# 🔐 Sauvegarde dans le dossier choisi
chemin_sauvegarde = r"C:\INSEEDS\PROJET AFTER BAC\random_forest_orientation.pkl"
joblib.dump(best_model, chemin_sauvegarde)

print(f"✅ Modèle sauvegardé avec succès dans : {chemin_sauvegarde}")


#___________________________________________


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


























