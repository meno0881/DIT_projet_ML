**déploiement d'un modèle IA de détection de fraude bancaire avec Streamlit**

# Objectifs 

-	Structurer un projet de machine learning en vue d'un déploiement
-	Entraîner et sérialiser un modèle de détection de fraude bancaire
-	Construire une interface web interactive avec Streamlit
-	Déployer l'application sur Streamlit Community Cloud
-	Appliquer les bonnes pratiques de mise en production (sécurité, monitoring, gestion des versions)

# Pre-requis
Élément	                Détail
Langage	                Python 3.9+
Bibliothèques	          pandas, scikit-learn, streamlit, joblib, matplotlib


# Structure du projet
Une structure claire facilite la maintenance et le déploiement :

DIT-projet-ML/

│
├── data/
│   └── transactions.csv          # Jeu de données (ex: Kaggle Credit Card Fraud)
│
├── model/
│   ├── train_model.py            # Script d'entraînement
│   └── fraud_model.pkl           # Modèle sérialisé (généré)
        EDA.ipynb                 # script EDA
│
├── app.py                        # Application Streamlit
├── requirements.txt              # Dépendances
├── .gitignore


# application streamlit 

lien: ditprojetml-kvnbihyzww8an6wqdfcsxu



└── README.md
