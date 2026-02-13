import joblib
import numpy as np

# Les colonnes de nos datasets
FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

OCEAN_CATEGORIES = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

model = None
scaler = None


def load_model():
    global model, scaler
    model = joblib.load("./models/my_model_v2.pkl")
    scaler = joblib.load("./models/scaler.pkl")


def predict_house_value(longitude, latitude, housing_median_age,
                        total_rooms, total_bedrooms, population,
                        households, median_income, ocean_proximity):
    
    # Calcul features engineered
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    # One-hot encoding pour ocean_proximity(plus de label du coup, comme la version 1...)
    ocean_encoded = [1 if cat == ocean_proximity else 0 for cat in OCEAN_CATEGORIES]

    # Construction du tableau des features dans le bon ordre et utilisation du déballeur *ocean_encoded->
    features = np.array([[
        longitude, latitude, housing_median_age,
        total_rooms, total_bedrooms, population,
        households, median_income,
        *ocean_encoded,
        rooms_per_household, bedrooms_per_room, population_per_household
    ]])

    # Normalisation puis prédiction
    features_scaled = scaler.transform(features)
    nos_prediction = model.predict(features_scaled)

    # arrondie de la prédiction à la 2ième
    return round(nos_prediction[0], 2)