# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Définition de la salle cible pour la simulation
target_room = "F106"

# Fichier CSV contenant les données simulées
csv_file = "simulation_bureau2.csv"
last_timestamp = None  # Variable pour stocker le dernier timestamp traité

# Génération de données factices pour l'entraînement du modèle
np.random.seed(42)  # Fixer la graine pour la reproductibilité
n_samples = 500  # Nombre de samples à générer
data_train = {
    'temperature_exterieure': np.random.uniform(-5, 35, n_samples),  # Température extérieure aléatoire
    'humidite': np.random.uniform(20, 80, n_samples),  # Humidité aléatoire
    'heure': np.random.randint(0, 24, n_samples),  # Heure aléatoire
    'aeration': np.random.choice([0, 1], n_samples),  # Aération (0 ou 1)
    'occupation': np.random.randint(0, 20, n_samples),  # Nombre d'occupants (aléatoire)
    'habitude_utilisateur': np.random.uniform(18, 24, n_samples),  # Habitude de température de l'utilisateur
    'contrainte_energie': np.random.uniform(0.5, 1.5, n_samples)  # Contrainte énergétique aléatoire
}

# Calcul de la température optimale en fonction des habitudes et de la contrainte énergétique
data_train["temperature_optimale"] = (
    data_train['habitude_utilisateur'] * data_train['contrainte_energie'] +  # Température optimale calculée
    np.random.normal(0, 1, n_samples)  # Ajouter du bruit à la température optimale
)
df_train = pd.DataFrame(data_train)  # Conversion des données en DataFrame pandas

# Définition des variables X (features) et y (cible)
X = df_train.drop(columns=["temperature_optimale"])  # Suppression de la colonne cible de X
y = df_train["temperature_optimale"]  # La cible est la température optimale

# Création et entraînement du modèle RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 arbres dans la forêt
model.fit(X, y)  # Entraînement du modèle avec les données

# Création de la figure pour le graphique
plt.ion()  # Mode interactif activé pour mettre à jour le graphique en temps réel
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Temps")
ax.set_ylabel("Température (°C)")
ax.set_title("Température Prédite vs Température Actuelle")
line_pred, = ax.plot([], [], label="Température Prédite", color="blue")
line_act, = ax.plot([], [], label="Température Actuelle", color="red")
ax.legend()

# Listes pour stocker les valeurs de température pour le graphique
times = []
temp_pred_values = []
temp_act_values = []

# Boucle infinie pour traiter les données en temps réel
while True:
    # Lire le fichier CSV et convertir la colonne "timestamp" en format datetime
    df = pd.read_csv(csv_file, parse_dates=["timestamp"])

    # Filtrer les données pour ne garder que celles de la salle cible
    if target_room:
        df = df[df["room"] == target_room]

    # Vérifier si de nouvelles données sont arrivées (par rapport à la dernière ligne traitée)
    if last_timestamp is None or df["timestamp"].max() > last_timestamp:
        last_timestamp = df["timestamp"].max()  # Mettre à jour le dernier timestamp
        new_data = df[df["timestamp"] == last_timestamp]  # Extraire les nouvelles données

        # Regrouper les données par timestamp pour calculer des moyennes et des sommes
        grouped = new_data.groupby("timestamp").agg({
            "temp_ext": "mean",  # Moyenne de la température extérieure
            "temp_int": "mean",  # Moyenne de la température intérieure
            "window_open": "max",  # Maximum pour l'état de la fenêtre (ouverte ou fermée)
            "presence": "sum"  # Somme des présences (combien de personnes présentes)
        }).reset_index()

        # Renommer les colonnes pour correspondre aux noms attendus par le modèle
        grouped.rename(columns={"window_open": "aeration", "presence": "occupation"}, inplace=True)
        
        # Ajouter des colonnes supplémentaires nécessaires pour la prédiction
        grouped["humidite"] = 50  # Humidité constante de 50 pour la simulation
        grouped["heure"] = grouped["timestamp"].dt.hour  # Extraire l'heure du timestamp
        grouped["habitude_utilisateur"] = grouped["temp_int"]  # Température intérieure comme habitude utilisateur
        grouped["contrainte_energie"] = 1.0  # Valeur fixe pour la contrainte énergétique

        # Sélectionner la dernière ligne de données pour la prédiction
        nouvelle_donnee = grouped.iloc[-1]

        # Préparer les caractéristiques pour faire la prédiction
        features = np.array([[
            nouvelle_donnee["temp_ext"],
            nouvelle_donnee["humidite"],
            nouvelle_donnee["heure"],
            nouvelle_donnee["aeration"],
            nouvelle_donnee["occupation"],
            nouvelle_donnee["habitude_utilisateur"],
            nouvelle_donnee["contrainte_energie"]
        ]])

        # Effectuer la prédiction de la température optimale
        prediction = model.predict(features)[0]
        temp_actuelle = nouvelle_donnee["temp_int"]  # Température intérieure actuelle

        # Afficher les résultats de la prédiction et de la température actuelle
        print(f"\n📊 Nouvelle donnée reçue à {last_timestamp}:")
        print(f"Température optimale prédite : {prediction:.2f}°C")
        print(f"Température actuelle : {temp_actuelle:.2f}°C")
        
        #Conditions pour augementer ou réduire le chauffage
        if (temp_actuelle > prediction):
            print("Réduire le Chauffage")
        elif (temp_actuelle < prediction):
            print("Augmenter le Chauffage")
        else :
            print("Température optimale atteinte")
        # Si l'écart entre la température actuelle et la prédite est trop grand, afficher une alerte
        if abs(temp_actuelle - prediction) > 5:
            print("⚠️ Alerte : Température anormale détectée !")
            
        # Ajouter les nouvelles données aux listes pour le tracé
        times.append(last_timestamp)
        temp_pred_values.append(prediction)
        temp_act_values.append(temp_actuelle)

        # Mettre à jour le graphique avec les nouvelles données
        line_pred.set_xdata(times)
        line_pred.set_ydata(temp_pred_values)
        line_act.set_xdata(times)
        line_act.set_ydata(temp_act_values)

        # Réactualiser le graphique
        plt.draw()
        plt.pause(0.1)  # Pause pour permettre le rafraîchissement de la fenêtre

    # Attendre 2 secondes avant de vérifier à nouveau
    time.sleep(2)  # Vérifier toutes les 2 secondes
