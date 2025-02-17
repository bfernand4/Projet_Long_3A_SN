import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import time  # Ajouter time pour la pause entre les écritures
import matplotlib.pyplot as plt
import csv

# ----- Paramètres de la simulation -----
# Ici, nous simulons les données de plusieurs salles sans en filtrer une en particulier.
# L'idée est d'avoir un ensemble de données réaliste sur les habitudes des employés.

# Définition des profils types de personnes (avec plages horaires et préférences de température)
profiles = [
    {"name": "Personne Matinale", "arrival": (7, 9), "departure": (15, 17), "temp_pref": (20, 22)},
    {"name": "Personne Standard", "arrival": (8, 10), "departure": (17, 19), "temp_pref": (21, 23)},
    {"name": "Personne Tardive", "arrival": (10, 12), "departure": (19, 21), "temp_pref": (22, 24)},
    {"name": "Aléatoire", "arrival": (7, 12), "departure": (15, 21), "temp_pref": (20, 24)},
]

# Fixer la graine aléatoire pour garantir la reproductibilité des résultats
random.seed(42)
np.random.seed(42)

# Liste de prénoms pour générer des utilisateurs uniques
names = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin", "Liam"]

# Génération aléatoire du nombre de personnes par profil (entre 3 et 6 personnes par type)
num_users_per_profile = {p["name"]: random.randint(3, 6) for p in profiles}

# Liste des salles disponibles (F100 à F109)
rooms = [f"F10{x}" for x in range(10)]

# Générer des utilisateurs uniques avec attribution d'une salle aléatoire
users = []
for profile in num_users_per_profile:
    count = num_users_per_profile[profile]
    for i in range(count):
        user_name = f"{random.choice(names)}_{i}"  # Attribution d'un prénom aléatoire
        user_room = random.choice(rooms)  # Attribution aléatoire d'une salle
        users.append({"user_id": user_name, "profile": profile, "room": user_room})

# On simule les données pour tous les utilisateurs
room_users = users

# Définition de la période de simulation (du 1er février au 1er mars 2024)
start_date = datetime(2024, 2, 1)
end_date = datetime(2024, 3, 1)
time_step = timedelta(minutes=10)  # Intervalle de 10 minutes entre chaque point de données

data = []  # Liste pour stocker les données simulées
current_time = start_date  # Initialisation du temps

# Ouverture du fichier CSV en mode 'w' pour l'écriture (création du fichier s'il n'existe pas)
with open("simulation_bureau2.csv", "w", newline='') as f:
    # Définir les colonnes du fichier CSV
    columns = [
        "timestamp", "user_id", "room", "presence", "temp_int", "temp_ext",
        "window_open", "heating_on", "cooling_on"
    ]
    writer = csv.writer(f)
    writer.writerow(columns)  # Écrire l'entête du CSV

    while current_time < end_date:
        for user in room_users:
            # Récupération du profil utilisateur
            profile = next(p for p in profiles if p["name"] == user["profile"])
            
            # Génération d'une heure d'arrivée et de départ aléatoire dans la plage définie par le profil
            arrival_hour = random.randint(*profile["arrival"])
            departure_hour = random.randint(*profile["departure"])
            temp_pref = random.uniform(*profile["temp_pref"])  # Préférence de température

            # Création des horaires de présence de base
            arrival_time = current_time.replace(hour=arrival_hour, minute=0)
            departure_time = current_time.replace(hour=departure_hour, minute=0)

            # Simulation de pauses :
            # - Pause repas entre 12h et 14h (50% de chance)
            # - Pause café aléatoire (10% de chance)
            pause = False
            if 12 <= current_time.hour < 14 and random.random() < 0.5:
                pause = True
            if random.random() < 0.1:
                pause = True

            # Détermination de la présence en fonction de l'horaire et des pauses
            if pause:
                presence = 0
            else:
                presence = 1 if arrival_time <= current_time <= departure_time else 0

            # Simulation de la température intérieure en fonction de la présence et des préférences utilisateur
            if presence:
                temp_int = temp_pref + random.uniform(-1, 1)
            else:
                temp_int = random.uniform(16, 20)  # Valeur par défaut en cas d'absence

            # Simulation de la température extérieure (variation en fonction de l'heure)
            hour = current_time.hour
            temp_ext = random.uniform(5, 15) if 6 <= hour <= 18 else random.uniform(-2, 7)

            # Probabilité d'ouverture de la fenêtre : 10% si présent, 5% sinon
            window_open = 1 if (random.random() < (0.1 if presence else 0.05)) else 0

            # Influence de l'ouverture de la fenêtre sur la température intérieure
            alpha = 0.3 if window_open else 0.1
            temp_int = temp_int + alpha * (temp_ext - temp_int)

            # Chauffage/climatisation activé en fonction de la différence avec la température préférée
            heating_on = 1 if presence and temp_int < temp_pref - 1 else 0
            cooling_on = 1 if presence and temp_int > temp_pref + 1 else 0

            # Enregistrement des données sous forme de liste
            data_row = [
                current_time,
                user["user_id"],
                user["room"],
                presence,
                temp_int,
                temp_ext,
                window_open,
                heating_on,
                cooling_on
            ]
            writer.writerow(data_row)  # Écriture des données dans le fichier CSV

        current_time += time_step  # Passage à la prochaine période de 10 minutes

        time.sleep(0.1)  # Pause de 0,1 seconde avant de générer la prochaine ligne

