# Fraud Detection — End-to-End ML System

##  Problème métier
Détection en temps réel de transactions par carte frauduleuses afin de
minimiser les pertes financières, tout en limitant les fausses alertes
qui dégradent l'expérience client.

**Pourquoi c'est difficile :**
- Fort déséquilibre des classes (~3,5 % de fraude)
- Coût asymétrique des erreurs (fraude ratée vs. client bloqué à tort)
- Variables anonymisées + structure temporelle

##  Données
IEEE-CIS Fraud Detection (Kaggle) — ~590 000 transactions, label `isFraud`.
Fusion de `train_transaction` et `train_identity` sur `TransactionID`.

##  Approche
1. **Modèle** — Gradient boosting (XGBoost/LightGBM) sur données tabulaires
2. **Explicabilité** — SHAP (contributions par variable)
3. **Déploiement** — API FastAPI + Docker sur AWS
4. **Interface** — Dashboard Streamlit (test + suivi)
5. **Monitoring** — Détection de dérive des données

##  Architecture
(TODO : ajouter le schéma d'architecture)

##  Résultats
(TODO : AUC, PR-AUC, métrique de coût métier, matrice de confusion)

##  Démo en ligne
(TODO : lien API + lien dashboard)

##  Stack technique
Python · pandas · scikit-learn · XGBoost · SHAP · MLflow · FastAPI
· Docker · AWS · Streamlit

## 📂 Structure du projet

```text
fraud-detection/
│
├── data/                       # Données (ignoré par git)
│   ├── raw/                    # CSV bruts de Kaggle
│   └── processed/              # Données nettoyées / features
│
├── notebooks/                  # Exploration et prototypage
│   ├── 01_eda.ipynb            # Analyse exploratoire
│   ├── 02_features.ipynb       # Feature engineering
│   └── 03_modeling.ipynb       # Entraînement / comparaison de modèles
│
├── src/                        # Code source réutilisable
│   ├── config.py               # Chemins, constantes, paramètres
│   ├── data.py                 # Chargement + fusion des données
│   ├── features.py             # Construction des features
│   ├── train.py                # Pipeline d'entraînement + MLflow
│   ├── predict.py              # Logique de prédiction + SHAP
│   └── api.py                  # API FastAPI
│
├── dashboard/                  # Interface Streamlit
│   └── app.py
│
├── models/                     # Modèles entraînés (ignoré par git)
├── tests/                      # Tests unitaires
├── docker/                     # Conteneurisation
│   └── Dockerfile
│
├── .github/workflows/          # CI/CD (GitHub Actions)
│   └── ci.yml
│
├── requirements.txt            # Dépendances Python
├── .gitignore
├── .env.example                # Modèle de variables d'environnement
└── README.md
```
## Récupération des données

Les données ne sont pas versionnées (trop volumineuses). Pour les récupérer :

```bash
# Nécessite un compte Kaggle + un token API (~/.kaggle/kaggle.json)
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

##  Installation & utilisation
(TODO : à compléter après la mise en place de l'API)