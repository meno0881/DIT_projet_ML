import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Détection Fraude", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    model      = joblib.load("model/fraud_model.pkl")
    scaler     = joblib.load("model/scaler.pkl")
    le_type    = joblib.load("model/le_type.pkl")
    le_status  = joblib.load("model/le_status.pkl")
    le_ville   = joblib.load("model/le_ville.pkl")
    target_map = joblib.load("model/target_map.pkl")
    return model, scaler, le_type, le_status, le_ville, target_map

model, scaler, le_type, le_status, le_ville, target_map = load_artifacts()
target_labels = {v:k for k,v in target_map.items()}

st.title("🏦 Système de Détection de Fraude Bancaire")
st.caption("Random Forest — 3 classes : Normal / Suspect / Fraude")
st.divider()

mode = st.sidebar.radio("Mode", ["Transaction unique","Fichier CSV (lot)"])

# ════════════ MODE 1 : Transaction unique ════════════
if mode == "Transaction unique":
    st.subheader("Saisie manuelle")
    c1, c2, c3 = st.columns(3)
    with c1:
        montant  = st.number_input("Montant (FCFA)", min_value=0.0, value=50000.0)
        heure    = st.slider("Heure", 0, 23, 12)
    with c2:
        type_t   = st.selectbox("Type de transaction", le_type.classes_)
        status   = st.selectbox("Statut opération",    le_status.classes_)
    with c3:
        ville    = st.selectbox("Localisation",        le_ville.classes_)
        jour_sem = st.slider("Jour semaine (0=Lun)",   0, 6, 1)

    if st.button("Analyser", type="primary"):
        heure_risque = 1 if (heure>=22 or heure<=8) else 0
        feats = np.array([[
            montant, np.log1p(montant), heure, heure_risque,
            jour_sem, 15,
            le_type.transform([type_t])[0],
            le_status.transform([status])[0],
            le_ville.transform([ville])[0]
        ]])
        pred   = model.predict(scaler.transform(feats))[0]
        probas = model.predict_proba(scaler.transform(feats))[0]

        st.divider()
        if pred == 0:
            st.success(f"✅ NORMALE — Confiance : {probas[0]:.1%}")
        elif pred == 1:
            st.warning(f"⚠️ SUSPECTE — Confiance : {probas[1]:.1%}")
        else:
            st.error(f"🚨 FRAUDE DÉTECTÉE — Confiance : {probas[2]:.1%}")

        st.markdown("**Probabilités par classe :**")
        for cls, prob in zip(["Normal","Suspect","Fraude"], probas):
            st.write(f"{cls} : {prob:.1%}")
            st.progress(float(prob))

# ════════════ MODE 2 : Fichier CSV ════════════
else:
    st.subheader("Analyse par lot")
    fichier = st.file_uploader("Déposer transactions.csv", type=["csv"])
    if fichier:
        df = pd.read_csv(fichier, sep=";")
        st.write(f"{df.shape[0]} transactions chargées")
        st.dataframe(df.head(3))

        if st.button("Lancer l'analyse", type="primary"):
            df["Date"]         = pd.to_datetime(df["Date"])
            df["Heure"]        = df["Date"].dt.hour
            df["JourSemaine"]  = df["Date"].dt.dayofweek
            df["JourMois"]     = df["Date"].dt.day
            df["Heure_risque"] = df["Heure"].apply(lambda h:1 if h>=22 or h<=8 else 0)
            df["Montant_log"]  = np.log1p(df["Montant"])
            df["Type_enc"]     = le_type.transform(df["Type de transaction"])
            df["Status_enc"]   = le_status.transform(df["Status operation"])
            df["Ville_enc"]    = le_ville.transform(df["Localisation"])

            FEATURES = ["Montant","Montant_log","Heure","Heure_risque",
                        "JourSemaine","JourMois","Type_enc","Status_enc","Ville_enc"]
            X_sc = scaler.transform(df[FEATURES])
            df["Prediction"]   = model.predict(X_sc)
            df["Résultat"]     = df["Prediction"].map(target_labels)
            df["Proba_fraude"] = model.predict_proba(X_sc)[:,2]

            c1,c2,c3 = st.columns(3)
            c1.metric("✅ Normal",  int((df["Prediction"]==0).sum()))
            c2.metric("⚠️ Suspect", int((df["Prediction"]==1).sum()))
            c3.metric("🚨 Fraude",  int((df["Prediction"]==2).sum()))

            def style_row(row):
                if row["Résultat"]=="Fraude":
                    return ["background-color:#ffcccc"]*len(row)
                if row["Résultat"]=="Suspect":
                    return ["background-color:#fff3cd"]*len(row)
                return [""]*len(row)

            cols_aff=["ID Clients","Montant","Type de transaction",
                      "Localisation","Résultat","Proba_fraude"]
            st.dataframe(df[cols_aff].style.apply(style_row,axis=1))

            st.download_button("📥 Télécharger", df.to_csv(index=False).encode(),
                               "resultats.csv","text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("Module IA appliquée — Zone CEMAC")
