import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Chargement des données simulées depuis le CSV
df = pd.read_csv("simulation_bureau.csv", parse_dates=["timestamp"])

# 2. Agrégation des mesures par timestamp
#    Pour chaque timestamp, on prend la moyenne de la température extérieure/intérieure,
#    on considère qu'une fenêtre ouverte = aération active (max), et on somme les présences (occupation)
grouped = df.groupby("timestamp").agg({
    "temp_ext": "mean",
    "temp_int": "mean",
    "window_open": "max",  # Si une fenêtre est ouverte, aération active
    "presence": "sum"      # Nombre total de personnes présentes à ce timestamp
}).reset_index()

# Renommer window_open en aeration pour la suite
grouped.rename(columns={"window_open": "aeration", "presence": "occupation"}, inplace=True)

# 3. Création des variables attendues pour le modèle
#    - La température extérieure est déjà présente
#    - On fixe une humidité (ex : 50%) car elle n'est pas simulée
#    - L'heure est extraite du timestamp
#    - La température "habituelle" de réglage est estimée ici par la température intérieure moyenne mesurée
#    - On fixe un indice de contrainte énergétique (par exemple 1.0)
grouped["humidite"] = 50
grouped["heure"] = pd.to_datetime(grouped["timestamp"]).dt.hour
grouped["habitude_utilisateur"] = grouped["temp_int"]
grouped["contrainte_energie"] = 1.0

# Pour vérification, affichons quelques lignes de données agrégées
print("Aperçu des données agrégées :")
print(grouped.head())

# 4. Création d'un dataset d'entraînement synthétique
#    (ceci représente une référence pour que le modèle "apprenne" la relation entre
#     les features et la température optimale)
np.random.seed(42)
n_samples = 500
data_train = {
    'temperature_exterieure': np.random.uniform(-5, 35, n_samples),
    'humidite': np.random.uniform(20, 80, n_samples),
    'heure': np.random.randint(0, 24, n_samples),
    'aeration': np.random.choice([0, 1], n_samples),
    'occupation': np.random.randint(0, 20, n_samples),
    'habitude_utilisateur': np.random.uniform(18, 24, n_samples),
    'contrainte_energie': np.random.uniform(0.5, 1.5, n_samples)
}
# La température optimale est fonction des habitudes et de la contrainte énergétique,
# avec un bruit aléatoire ajouté.
data_train["temperature_optimale"] = (
    data_train['habitude_utilisateur'] * data_train['contrainte_energie'] +
    np.random.normal(0, 1, n_samples)
)
df_train = pd.DataFrame(data_train)

# 5. Séparation train/test et entraînement du modèle
X = df_train.drop(columns=["temperature_optimale"])
y = df_train["temperature_optimale"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nErreur absolue moyenne (MAE) du modèle : {mae:.2f}°C\n")

# 6. Sélection d'une nouvelle donnée pour la prédiction
#    Ici, nous prenons la dernière mesure agrégée comme exemple
nouvelle_donnee = grouped.iloc[-1]
print("Nouvelle donnée agrégée :")
print(nouvelle_donnee)

# Création du vecteur de features (dans l'ordre attendu par le modèle)
features = np.array([[
    nouvelle_donnee["temp_ext"],         # température_exterieure
    nouvelle_donnee["humidite"],           # humidité
    nouvelle_donnee["heure"],              # heure
    nouvelle_donnee["aeration"],           # aération
    nouvelle_donnee["occupation"],         # occupation
    nouvelle_donnee["habitude_utilisateur"],  # habitude_utilisateur
    nouvelle_donnee["contrainte_energie"]  # contrainte_energie
]])

# Température actuelle mesurée (moyenne de temp_int pour ce timestamp)
temp_actuelle = nouvelle_donnee["temp_int"]

# 7. Prédiction de la température optimale
prediction = model.predict(features)[0]
print(f"\nTempérature optimale prédite : {prediction:.2f}°C")
print(f"Température actuelle mesurée : {temp_actuelle:.2f}°C")

# 8. Vérification de l'écart et notification
seuil_alerte = 2  # Seuil d'alerte en °C
if abs(temp_actuelle - prediction) > seuil_alerte:
    print("⚠️ Alerte : La température actuelle est anormalement différente de la température optimale !")

# 9. Gestion de la présence et contrôle du chauffage
if nouvelle_donnee["occupation"] == 0:
    print("🏠 Personne dans la salle, chauffage éteint.")
elif temp_actuelle < prediction:
    print("🔥 Augmentation de la puissance du chauffage.")
elif temp_actuelle > prediction:
    print("❄️ Réduction de la puissance du chauffage.")
else:
    print("✅ Température stable, aucune action nécessaire.")
