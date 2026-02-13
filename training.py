import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from model import FEATURE_COLUMNS, load_model


def retrain_model(csv_path):
    #but: Réentraîner le modèle avec un nouveau CSV et retourne les score comme demandé

    #  Charger les données
    df = pd.read_csv(csv_path)

    #  Feature engineering
    df["rooms_per_household"] = df.total_rooms / df.households
    df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
    df["population_per_household"] = df.population / df.households

    #  One-hot encoding
    df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)

    # important! au cas ou il y a un manque de colonnes dans les autres datasets(donc faut mettre des données temps dummy)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    #  Split et train test
    X = df[FEATURE_COLUMNS]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normalisation ou standardisation
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    #  Entraînement
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_std, y_train)

    #  nos différents scores(juste 2 ici)
    y_pred = model.predict(X_test_std)
    r2 = round(r2_score(y_test, y_pred), 2)
    rmse = round(root_mean_squared_error(y_test, y_pred), 2)

    #  Export du nouveau modèle + scaler
    joblib.dump(model, "my_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    load_model()

    return {
        'r2': r2,'rmse': rmse,'n_train': len(X_train),'n_test': len(X_test),'n_features': X_train.shape[1],
    }