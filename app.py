import os
from flask import Flask, render_template, request, current_app
from model import load_model, predict_house_value
from training import retrain_model
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle au démarrage
load_model()

# Charger le dataframe au démarrage (accessible partout via current_app)
app.config['dataframe'] = pd.read_csv('data/housing_1.csv')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/Training', methods=['GET', 'POST'])
def Training():
    scores = None
    error = None

    if request.method == 'POST':
        file = request.files.get('dataset')

        if not file or file.filename == '':
            error = "Aucun fichier sélectionné."
        elif not file.filename.endswith('.csv'):
            error = "Le fichier doit être un .csv"
        else:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            try:
                scores = retrain_model(filepath)
                app.config['dataframe'] = pd.read_csv(filepath)
            except Exception as e:
                error = f"Erreur lors du réentraînement : {str(e)}"

    return render_template("training.html", scores=scores, error=error)


def make_zone_dropdown(fig, df, zones, x_col, y_col, chart_type="scatter", **kwargs):
    """créeation graphique avec dropdown filtre zone."""

    if chart_type == "scatter":
        # Trace "Toutes" avec tous les points
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='markers',
            marker=dict(
                color=df.get(kwargs.get('color_col', y_col)),
                colorscale=kwargs.get('colorscale', 'jet'),
                opacity=kwargs.get('opacity', 0.3),
                colorbar=dict(title=kwargs.get('colorbar_title', '')),
            ),
            name="Toutes",
            showlegend=False,
            visible=True
        ))
        # Une trace par zone
        for zone in zones:
            df_z = df[df['ocean_proximity'] == zone]
            fig.add_trace(go.Scatter(
                x=df_z[x_col], y=df_z[y_col],
                mode='markers',
                marker=dict(
                    color=df_z.get(kwargs.get('color_col', y_col)),
                    colorscale=kwargs.get('colorscale', 'jet'),
                    opacity=kwargs.get('opacity', 0.3),
                    showscale=False,
                ),
                name=zone,
                showlegend=False,
                visible=False
            ))

    # Boutons dropdown
    n = len(zones)
    buttons = [dict(label="Toutes", method="update",
                    args=[{"visible": [True] + [False]*n}])]
    for i in range(n):
        vis = [False] + [False]*n
        vis[i+1] = True
        buttons.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )
    return fig


@app.route('/Dashboard', methods=['GET', 'POST'])
def dashboard():
    df = current_app.config['dataframe']
    zones = sorted(df['ocean_proximity'].unique().tolist())

    # --- KPIs ---
    kpis = {
        'prix_median': f"${df['median_house_value'].median():,.0f}",
        'revenu_median': f"{df['median_income'].mean():.2f}",
        'nb_districts': f"{len(df):,}",
        'zone_chere': df.groupby('ocean_proximity')['median_house_value'].median().idxmax(),
    }

    # ============================================================
    # GRAPHIQUE 1 : Carte Mapbox + filtre zone
    # ============================================================
    fig_mapbox = go.Figure()

    # Trace "Toutes"
    fig_mapbox.add_trace(go.Scattermapbox(
        lat=df['latitude'], lon=df['longitude'],
        mode='markers',
        marker=dict(
            color=df['median_house_value'],
            colorscale='jet', opacity=0.5,
            size=5, colorbar=dict(title="Prix médian"),
        ),
        text=df['ocean_proximity'],
        hovertemplate="Zone: %{text}<br>Prix: %{marker.color:$,.0f}<extra></extra>",
        name="Toutes", showlegend=False, visible=True
    ))
    # Par zone
    for zone in zones:
        df_z = df[df['ocean_proximity'] == zone]
        fig_mapbox.add_trace(go.Scattermapbox(
            lat=df_z['latitude'], lon=df_z['longitude'],
            mode='markers',
            marker=dict(
                color=df_z['median_house_value'],
                colorscale='jet', opacity=0.5,
                size=5, showscale=False,
            ),
            text=df_z['ocean_proximity'],
            hovertemplate="Zone: %{text}<br>Prix: %{marker.color:$,.0f}<extra></extra>",
            name=zone, showlegend=False, visible=False
        ))

    n = len(zones)
    buttons_map = [dict(label="Toutes", method="update",
                        args=[{"visible": [True] + [False]*n}])]
    for i in range(n):
        vis = [False] + [False]*n
        vis[i+1] = True
        buttons_map.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig_mapbox.update_layout(
        title="Carte des prix immobiliers en Californie",
        mapbox=dict(style="open-street-map", zoom=5,
                    center=dict(lat=36.7, lon=-119.5)),
        height=600, margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            buttons=buttons_map, direction="down",
            showactive=True, x=0.01, xanchor="left", y=0.99, yanchor="top",
            bgcolor="white", bordercolor="#ccc"
        )]
    )

    # ============================================================
    # GRAPHIQUE 2 : Scatter 3D + filtre zone
    # ============================================================
    fig_3d = go.Figure()

    fig_3d.add_trace(go.Scatter3d(
        x=df['longitude'], y=df['latitude'], z=df['median_house_value'],
        mode='markers',
        marker=dict(color=df['median_house_value'], colorscale='jet',
                    opacity=0.4, size=2, colorbar=dict(title="Prix")),
        name="Toutes", showlegend=False, visible=True
    ))
    for zone in zones:
        df_z = df[df['ocean_proximity'] == zone]
        fig_3d.add_trace(go.Scatter3d(
            x=df_z['longitude'], y=df_z['latitude'], z=df_z['median_house_value'],
            mode='markers',
            marker=dict(color=df_z['median_house_value'], colorscale='jet',
                        opacity=0.4, size=2, showscale=False),
            name=zone, showlegend=False, visible=False
        ))

    buttons_3d = [dict(label="Toutes", method="update",
                       args=[{"visible": [True] + [False]*n}])]
    for i in range(n):
        vis = [False] + [False]*n
        vis[i+1] = True
        buttons_3d.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig_3d.update_layout(
        title="Vue 3D : Localisation × Prix",
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude",
                   zaxis_title="Prix médian"),
        height=600,
        updatemenus=[dict(
            buttons=buttons_3d, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.1, yanchor="top"
        )]
    )

    # ============================================================
    # GRAPHIQUE 3 : Scatter filtrable (existant) + filtre zone
    # ============================================================
    fig_map = go.Figure()

    for idx, zone in enumerate(zones):
        df_zone = df[df['ocean_proximity'] == zone]
        fig_map.add_trace(go.Scatter(
            x=df_zone['longitude'], y=df_zone['latitude'],
            mode='markers',
            marker=dict(
                color=df_zone['median_house_value'],
                colorscale='jet', opacity=0.3,
                colorbar=dict(title="Prix médian") if idx == 0 else None,
                showscale=(idx == 0),
            ),
            name=zone, text=df_zone['ocean_proximity'],
            visible=True, showlegend=False
        ))

    buttons_scatter = [dict(label="Toutes", method="update",
                            args=[{"visible": [True]*n}])]
    for i in range(n):
        vis = [False]*n
        vis[i] = True
        buttons_scatter.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig_map.update_layout(
        title="Prix médian par localisation (filtre par zone)",
        xaxis_title="Longitude", yaxis_title="Latitude", height=500,
        updatemenus=[dict(
            buttons=buttons_scatter, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    # ============================================================
    # GRAPHIQUE 4 : Revenu vs Prix + filtre zone
    # ============================================================
    fig_income = go.Figure()
    fig_income = make_zone_dropdown(
        fig_income, df, zones, "median_income", "median_house_value",
        color_col="median_house_value", colorbar_title="Prix", opacity=0.2
    )
    fig_income.update_layout(
        title="Revenu médian vs Prix médian",
        xaxis_title="Revenu médian", yaxis_title="Prix médian", height=400
    )

    # ============================================================
    # GRAPHIQUE 5 : Boxplot + filtre métrique
    # ============================================================
    metrics = {
        'median_house_value': 'Prix médian ($)',
        'median_income': 'Revenu médian',
        'housing_median_age': 'Âge médian',
        'population': 'Population'
    }
    fig_box = go.Figure()

    for i, (col, label) in enumerate(metrics.items()):
        for zone in zones:
            df_z = df[df['ocean_proximity'] == zone]
            fig_box.add_trace(go.Box(
                y=df_z[col], name=zone,
                visible=(i == 0),
                showlegend=False
            ))

    n_zones = len(zones)
    buttons_box = []
    for i, (col, label) in enumerate(metrics.items()):
        vis = [False] * (len(metrics) * n_zones)
        for j in range(n_zones):
            vis[i * n_zones + j] = True
        buttons_box.append(dict(
            label=label, method="update",
            args=[{"visible": vis}, {"yaxis.title.text": label}]
        ))

    fig_box.update_layout(
        title="Distribution par zone",
        yaxis_title="Prix médian ($)", height=400,
        updatemenus=[dict(
            buttons=buttons_box, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    # ============================================================
    # GRAPHIQUE 6 : Prix par zone + filtre moyenne/médiane
    # ============================================================
    avg_by_zone = df.groupby('ocean_proximity')['median_house_value'].mean().sort_values()
    med_by_zone = df.groupby('ocean_proximity')['median_house_value'].median().sort_values()

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=avg_by_zone.values, y=avg_by_zone.index,
        orientation='h', name="Moyenne", visible=True
    ))
    fig_bar.add_trace(go.Bar(
        x=med_by_zone.values, y=med_by_zone.index,
        orientation='h', name="Médiane", visible=False
    ))

    fig_bar.update_layout(
        title="Prix par zone", xaxis_title="Prix ($)", height=400, showlegend=False,
        updatemenus=[dict(
            buttons=[
                dict(label="Moyenne", method="update",
                     args=[{"visible": [True, False]},
                           {"title": "Prix moyen par zone"}]),
                dict(label="Médiane", method="update",
                     args=[{"visible": [False, True]},
                           {"title": "Prix médian par zone"}]),
            ],
            direction="down", showactive=True,
            x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    # ============================================================
    # GRAPHIQUE 7 : Âge vs Prix (density heatmap) + filtre zone
    # ============================================================
    fig_age = go.Figure()

    # Trace "Toutes" : density heatmap
    fig_age.add_trace(go.Histogram2d(
        x=df['housing_median_age'], y=df['median_house_value'],
        colorscale='YlOrRd', nbinsx=40, nbinsy=40,
        colorbar=dict(title="Nb districts"),
        visible=True, name="Toutes"
    ))
    # Par zone : density heatmap
    for zone in zones:
        df_z = df[df['ocean_proximity'] == zone]
        fig_age.add_trace(go.Histogram2d(
            x=df_z['housing_median_age'], y=df_z['median_house_value'],
            colorscale='YlOrRd', nbinsx=40, nbinsy=40,
            showscale=False,
            visible=False, name=zone
        ))

    buttons_age = [dict(label="Toutes", method="update",
                        args=[{"visible": [True] + [False]*n}])]
    for i in range(n):
        vis = [False] + [False]*n
        vis[i+1] = True
        buttons_age.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig_age.update_layout(
        title="Âge des logements vs Prix (densité)",
        xaxis_title="Âge médian du logement",
        yaxis_title="Prix médian ($)",
        height=500,
        updatemenus=[dict(
            buttons=buttons_age, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    # ============================================================
    # GRAPHIQUE 8 : Matrice de corrélation + filtre zone
    # ============================================================
    num_cols = df.drop(['id', 'ocean_proximity'], axis=1).columns.tolist()

    fig_corr = go.Figure()

    # Trace "Toutes"
    corr_all = df[num_cols].corr()
    fig_corr.add_trace(go.Heatmap(
        z=corr_all.values, x=num_cols, y=num_cols,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=np.round(corr_all.values, 2), texttemplate="%{text}",
        visible=True, name="Toutes"
    ))
    # Par zone
    for zone in zones:
        df_z = df[df['ocean_proximity'] == zone]
        corr_z = df_z[num_cols].corr()
        fig_corr.add_trace(go.Heatmap(
            z=corr_z.values, x=num_cols, y=num_cols,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr_z.values, 2), texttemplate="%{text}",
            visible=False, name=zone
        ))

    buttons_corr = [dict(label="Toutes", method="update",
                         args=[{"visible": [True] + [False]*n}])]
    for i in range(n):
        vis = [False] + [False]*n
        vis[i+1] = True
        buttons_corr.append(dict(label=zones[i], method="update", args=[{"visible": vis}]))

    fig_corr.update_layout(
        title="Matrice de corrélation",
        height=500,
        updatemenus=[dict(
            buttons=buttons_corr, direction="down",
            showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="top"
        )]
    )

    return render_template(
        "dashboard.html",
        kpis=kpis,
        fig_mapbox=fig_mapbox.to_html(full_html=False),
        fig_3d=fig_3d.to_html(full_html=False),
        fig_map=fig_map.to_html(full_html=False),
        fig_income=fig_income.to_html(full_html=False),
        fig_box=fig_box.to_html(full_html=False),
        fig_bar=fig_bar.to_html(full_html=False),
        fig_age=fig_age.to_html(full_html=False),
        fig_corr=fig_corr.to_html(full_html=False),
    )


@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    result = None

    if request.method == 'POST':
        longitude = float(request.form.get('longitude'))
        latitude = float(request.form.get('latitude'))
        housing_median_age = float(request.form.get('housing_median_age'))
        total_rooms = float(request.form.get('total_rooms'))
        total_bedrooms = float(request.form.get('total_bedrooms'))
        population = float(request.form.get('population'))
        households = float(request.form.get('households'))
        median_income = float(request.form.get('median_income'))
        ocean_proximity = request.form.get('ocean_proximity')

        result = predict_house_value(
            longitude, latitude, housing_median_age,
            total_rooms, total_bedrooms, population,
            households, median_income, ocean_proximity
        )

    return render_template("prediction.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
