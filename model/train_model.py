import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib, matplotlib.pyplot as plt

# ── 1. Chargement ────────────────────────────────────────────
df = pd.read_csv("data/transactions.csv", sep=";")

# ── 2. Features temporelles ──────────────────────────────────
df["Date"]         = pd.to_datetime(df["Date"])
df["Heure"]        = df["Date"].dt.hour
df["JourSemaine"]  = df["Date"].dt.dayofweek
df["JourMois"]     = df["Date"].dt.day

# Feature binaire : plage horaire à risque (pic fraude à 6h)
df["Heure_risque"] = df["Heure"].apply(lambda h: 1 if h>=22 or h<=8 else 0)

# ── 3. Transformation log du montant ─────────────────────────
# Réduit l'effet des valeurs extrêmes (1 000 → 15 000 000 FCFA)
df["Montant_log"] = np.log1p(df["Montant"])

# ── 4. Encodage de la variable cible ─────────────────────────
target_map = {"Normal": 0, "Suspect": 1, "Fraude": 2}
df["Target_num"] = df["Target"].map(target_map)

# ── 5. Encodage des variables catégorielles ──────────────────
le_type   = LabelEncoder()
le_status = LabelEncoder()
le_ville  = LabelEncoder()

df["Type_enc"]   = le_type.fit_transform(df["Type de transaction"])
df["Status_enc"] = le_status.fit_transform(df["Status operation"])
df["Ville_enc"]  = le_ville.fit_transform(df["Localisation"])

# Sauvegarde des encodeurs pour Streamlit
joblib.dump(le_type,    "model/le_type.pkl")
joblib.dump(le_status,  "model/le_status.pkl")
joblib.dump(le_ville,   "model/le_ville.pkl")
joblib.dump(target_map, "model/target_map.pkl")
print("Encodeurs sauvegardés.")
# ── 6. Sélection des features et split ──────────────────────
FEATURES = [
    "Montant", "Montant_log",
    "Heure", "Heure_risque", "JourSemaine", "JourMois",
    "Type_enc", "Status_enc", "Ville_enc"
]

X = df[FEATURES]
y = df["Target_num"]

print(f"Distribution : {dict(y.value_counts())}")
# Attendu : {0: 4093, 1: 1091, 2: 198}

# stratify=y garantit la représentation des 3 classes dans train ET test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print(f"Fraudes dans test : {(y_test==2).sum()}")
# ── 7. Normalisation ─────────────────────────────────────────
# StandardScaler : ramène chaque feature à (moyenne=0, écart-type=1)

scaler = StandardScaler()

# fit_transform sur train uniquement → le scaler "apprend" les paramètres
X_train_scaled = scaler.fit_transform(X_train)

# transform seulement sur test → on applique les paramètres appris
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")
print("Scaler sauvegardé.")
# ── 8. RandomForestClassifier ────────────────────────────────
model = RandomForestClassifier(
    n_estimators  = 200,        # 200 arbres de décision
    max_depth     = 10,         # profondeur maximale par arbre
    class_weight  = "balanced", # compense le déséquilibre des 3 classes
    random_state  = 42,         # reproductibilité
    n_jobs        = -1          # parallélisation tous CPU
)

model.fit(X_train_scaled, y_train)
print("Modèle entraîné.")

joblib.dump(model, "model/fraud_model.pkl")
print("Modèle sauvegardé : model/fraud_model.pkl")
# ── 9. Prédictions ───────────────────────────────────────────
y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# ── 10. Rapport de classification (métrique principale) ───────
print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal (0)","Suspect (1)","Fraude (2)"]
))

# ── 11. Matrice de confusion ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal","Suspect","Fraude"]
)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matrice de confusion — 3 classes")
plt.tight_layout()
plt.savefig("model/captures/06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 12. Importance des variables ─────────────────────────────
importances = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=True)

importances.plot(kind="barh", color="#2E86C1", figsize=(9,6))
plt.title("Importance des variables — Random Forest")
plt.xlabel("Score d'importance (Gini)")
plt.tight_layout()
plt.savefig("model/captures/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== TOP FEATURES ===")
print(importances.sort_values(ascending=False))
