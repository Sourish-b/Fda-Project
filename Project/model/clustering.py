import os

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def _resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


GEOJSON_STATE_FIXES = {
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Delhi": "NCT of Delhi",
    "Andaman and Nicobar": "Andaman And Nicobar Islands",
    "Jammu and Kashmir": "Jammu And Kashmir",
    "Pondicherry": "Puducherry"
}


def run_clustering(df_scaled, df_states):
    """Run KMeans clustering, save diagnostics, and return labeled state DataFrame."""
    if len(df_scaled) < 2:
        raise ValueError("df_scaled must contain at least 2 rows for KMeans(n_clusters=2).")

    states = df_states.copy()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)

    max_k = min(10, len(df_scaled))
    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(df_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "elbow_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(df_scaled)
    states["Cluster"] = cluster_ids

    state_col = _resolve_column(states, ["Name of State/UT", "name_of_state_ut"])
    wind_col = _resolve_column(states, ["Wind", "wind"])
    solar_col = _resolve_column(states, ["Solar", "solar"])
    biomass_col = _resolve_column(states, ["Biomass", "biomass"])
    small_hydro_col = _resolve_column(states, ["Small Hydro", "small_hydro"])
    total_col = _resolve_column(states, ["Total_Renewable", "total_renewable"])

    if not all([state_col, wind_col, solar_col, biomass_col, small_hydro_col]):
        raise KeyError("Missing required columns. Needed: Name of State/UT, Solar, Wind, Biomass, Small Hydro")

    if total_col is None:
        states["Total_Renewable"] = (
            pd.to_numeric(states[wind_col], errors="coerce")
            + pd.to_numeric(states[solar_col], errors="coerce")
            + pd.to_numeric(states[biomass_col], errors="coerce")
            + pd.to_numeric(states[small_hydro_col], errors="coerce")
        )
        total_col = "Total_Renewable"

    mean_total = states.groupby("Cluster")[total_col].mean()
    energy_hub_cluster = mean_total.idxmax()

    cluster_name_map = {
        energy_hub_cluster: "Energy Hub",
        1 - energy_hub_cluster: "Energy Consumer",
    }
    states["Cluster"] = states["Cluster"].map(cluster_name_map)

    result_df = pd.DataFrame(
        {
            "Name of State/UT": states[state_col],
            "Cluster": states["Cluster"],
            "Total_Renewable": states[total_col],
            "Solar": states[solar_col],
            "Wind": states[wind_col],
            "Biomass": states[biomass_col],
            "Small Hydro": states[small_hydro_col],
        }
    )

    result_df.to_csv(os.path.join(save_dir, "cluster_labels.csv"), index=False)

    print("Cluster counts:")
    print(result_df["Cluster"].value_counts())
    print("Mean Total_Renewable per cluster:")
    print(result_df.groupby("Cluster")["Total_Renewable"].mean())

    return states


def plot_clusters(df_labeled):
    """Plot labeled state clusters and save a Total_Renewable vs ghi scatter chart."""
    data = df_labeled.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    total_col = _resolve_column(data, ["Total_Renewable", "total_renewable"])
    ghi_col = _resolve_column(data, ["ghi", "GHI"])
    cluster_col = _resolve_column(data, ["Cluster", "cluster"])

    if not all([state_col, total_col, ghi_col, cluster_col]):
        raise KeyError("Missing required columns. Needed: Name of State/UT, Total_Renewable, ghi, Cluster")

    color_map = {
        "Energy Hub": "green",
        "Energy Consumer": "red",
    }

    plt.figure(figsize=(10, 7))

    for cluster_name in ["Energy Hub", "Energy Consumer"]:
        cluster_df = data[data[cluster_col] == cluster_name]
        if cluster_df.empty:
            continue
        plt.scatter(
            cluster_df[total_col],
            cluster_df[ghi_col],
            c=color_map[cluster_name],
            label=cluster_name,
            alpha=0.8,
            s=60,
        )

    for _, row in data.iterrows():
        plt.annotate(
            str(row[state_col]),
            (row[total_col], row[ghi_col]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.xlabel("Total_Renewable")
    plt.ylabel("ghi")
    plt.title("State Clusters: Energy Hub vs Consumer")
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "cluster_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_choropleth(df_labeled):
    """Plot and save India choropleth for Energy Hub vs Energy Consumer states."""
    data = df_labeled.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    cluster_col = _resolve_column(data, ["Cluster", "cluster"])
    total_col = _resolve_column(data, ["Total_Renewable", "total_renewable"])

    if not all([state_col, cluster_col, total_col]):
        raise KeyError("Missing required columns. Needed: Name of State/UT, Cluster, Total_Renewable")

    geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    gdf = gpd.read_file(geojson_url)

    # 1. Update outdated map names to match your modern dataset
    gdf["NAME_1"] = gdf["NAME_1"].replace(GEOJSON_STATE_FIXES)

    # 2. Create temporary lowercase columns to ignore case sensitivity (like "and" vs "And")
    gdf["merge_name"] = gdf["NAME_1"].astype(str).str.strip().str.lower()
    data["merge_name"] = data[state_col].astype(str).str.strip().str.lower()

    # 3. Merge perfectly using the lowercase column
    merged = gdf.merge(data, on="merge_name", how="left")

    failed_states = sorted(set(data[state_col].dropna()) - set(merged[state_col].dropna()))
    print("States failed to merge:")
    if failed_states:
        for state in failed_states:
            print(state)
    else:
        print("None")

    def _fill_color(cluster_value):
        if cluster_value == "Energy Hub":
            return "#0ad25d"
        if cluster_value == "Energy Consumer":
            return "#d32e1b"
        return "gray"

    merged["fill_color"] = merged[cluster_col].apply(_fill_color)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged.plot(color=merged["fill_color"], ax=ax, edgecolor="black", linewidth=0.5)

    top5_hubs = (
        data[data[cluster_col] == "Energy Hub"]
        .sort_values(total_col, ascending=False)
        .head(5)[state_col]
        .tolist()
    )

    hub_geo = merged[merged["NAME_1"].isin(top5_hubs)].copy()
    for _, row in hub_geo.iterrows():
        if row.geometry is None:
            continue
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, row["NAME_1"], fontsize=8, ha="center", va="center")

    ax.set_title("India Renewable Energy Hubs")
    ax.set_axis_off()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "india_choropleth.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_folium_map(df_labeled):
    """Create and save an interactive Folium map for Energy Hub vs Consumer clusters."""
    data = df_labeled.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    cluster_col = _resolve_column(data, ["Cluster", "cluster"])
    total_col = _resolve_column(data, ["Total_Renewable", "total_renewable"])
    solar_col = _resolve_column(data, ["Solar", "solar"])
    wind_col = _resolve_column(data, ["Wind", "wind"])

    if not all([state_col, cluster_col, total_col, solar_col, wind_col]):
        raise KeyError(
            "Missing required columns. Needed: Name of State/UT, Cluster, Total_Renewable, Solar, Wind"
        )

    geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    gdf = gpd.read_file(geojson_url)

    # 1. Catch ALL map name mismatches
    gdf["NAME_1"] = gdf["NAME_1"].replace(GEOJSON_STATE_FIXES)

    gdf["merge_name"] = gdf["NAME_1"].astype(str).str.strip().str.lower()
    data["merge_name"] = data[state_col].astype(str).str.strip().str.lower()

    merged = gdf.merge(data, on="merge_name", how="left")

    merged["State Name"] = merged["NAME_1"]
    merged["Cluster"] = merged[cluster_col].fillna("Unmatched")
    merged["Total_Renewable"] = merged[total_col]
    merged["Solar"] = merged[solar_col]
    merged["Wind"] = merged[wind_col]

    # Center the map perfectly
    india_map = folium.Map(location=[22.5, 80.0], zoom_start=4.5)

    def _style_function(feature):
        cluster_value = feature["properties"].get("Cluster")
        if cluster_value == "Energy Hub":
            fill_color = "#00FF88" # Match Dark Mode Green
        elif cluster_value == "Energy Consumer":
            fill_color = "#FF3366" # Match Dark Mode Red
        else:
            fill_color = "#232D42" # Match Dark Mode Grey
            
        return {
            "fillColor": fill_color,
            "color": "#0B0F19",
            "weight": 1.5,
            "fillOpacity": 0.75,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["State Name", "Cluster", "Total_Renewable"],
        aliases=["State", "Type", "Total (MW)"],
        localize=True,
        sticky=False,
    )

    folium.GeoJson(
        data=merged.to_json(),
        style_function=_style_function,
        tooltip=tooltip,
        name="India Clusters",
    ).add_to(india_map)

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "india_map.html")
    india_map.save(output_path)

    # ---------------------------------------------------------
    # 2. INJECT CLICK EVENTS INTO THE HTML SO IT TALKS TO DASHBOARD
    # ---------------------------------------------------------
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    js_injection = """
    <script>
        // Wait 1 second for the map to fully load
        setTimeout(function() {
            for (var key in window) {
                if (key.startsWith('geo_json_')) {
                    var geojsonLayer = window[key];
                    geojsonLayer.eachLayer(function(layer) {
                        
                        // Make states feel clickable
                        if (layer.getElement && layer.getElement()) {
                            layer.getElement().style.cursor = 'pointer';
                        }

                        // Send the click event to the main dashboard!
                        layer.on('click', function(e) {
                            var stateName = e.target.feature.properties["State Name"];
                            if (window.parent && window.parent.highlightState) {
                                // Update charts and stats
                                window.parent.highlightState(stateName);
                                
                                // Update the dropdown box to match
                                var selectEl = window.parent.document.getElementById('stateSelect');
                                if(selectEl) {
                                    selectEl.value = stateName;
                                }
                            }
                        });
                    });
                }
            }
        }, 1000);
    </script>
    """
    # Insert the script right before the closing body tag
    html_content = html_content.replace("</body>", js_injection + "</body>")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)