import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from utils import find_state, normalize_state_name, order_months, safe_float


api_bp = Blueprint("api", __name__)


def _resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _state_col(df):
    return _resolve_column(df, ["Name of State/UT", "name_of_state_ut", "State", "state"])


def _cluster_col(df):
    return _resolve_column(df, ["Cluster", "cluster"])


def _total_col(df):
    return _resolve_column(df, ["Total_Renewable", "total_renewable"])


@api_bp.get("/states")
def get_states():
    try:
        df = current_app.config.get("cluster_df")
        if df is None:
            return jsonify({"error": "cluster_df not loaded"}), 500

        state_col = _state_col(df)
        cluster_col = _cluster_col(df)
        total_col = _total_col(df)
        if not all([state_col, cluster_col, total_col]):
            return jsonify({"error": "Required columns missing in cluster_df"}), 500

        out_df = df[[state_col, cluster_col, total_col]].copy()
        out_df[total_col] = pd.to_numeric(out_df[total_col], errors="coerce")
        out_df = out_df.sort_values(total_col, ascending=False)

        payload = [
            {
                "state": str(row[state_col]),
                "cluster": str(row[cluster_col]),
                "total_renewable": safe_float(row[total_col]),
            }
            for _, row in out_df.iterrows()
        ]
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.get("/state/<state_name>")
def get_state_profile(state_name):
    try:
        state_name = normalize_state_name(state_name)
        profiles = current_app.config.get("profiles_df")
        clusters = current_app.config.get("cluster_df")
        if profiles is None:
            return jsonify({"error": "profiles_df not loaded"}), 500

        state_col = _state_col(profiles)
        if state_col is None:
            return jsonify({"error": "State column missing in profiles_df"}), 500

        profile_row = find_state(profiles, state_name)
        if profile_row is None:
            return jsonify({"error": "State not found"}), 404

        row = profile_row

        if clusters is not None:
            c_state_col = _state_col(clusters)
            c_cluster_col = _cluster_col(clusters)
            if c_state_col and c_cluster_col:
                c_match = clusters[clusters[c_state_col].astype(str).str.lower() == state_name.lower()]
                if not c_match.empty:
                    row["Cluster"] = c_match.iloc[0][c_cluster_col]

        clean_row = {
            str(k): (None if pd.isna(v) else (float(v) if isinstance(v, (np.integer, np.floating, int, float)) else v))
            for k, v in row.items()
        }
        return jsonify(clean_row)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.get("/seasonal/<state_name>")
def get_state_seasonal(state_name):
    try:
        state_name = normalize_state_name(state_name)
        if state_name.lower() == "nct of delhi":
            state_name = "Delhi"
        df = current_app.config.get("seasonal_df")
        if df is None:
            return jsonify({"error": "seasonal_df not loaded"}), 500

        state_col = _state_col(df)
        month_col = _resolve_column(df, ["MONTH", "month"])
        wind_col = _resolve_column(df, ["Wind", "wind", "Installed - Wind Power", "installed_wind_power"])
        solar_col = _resolve_column(df, ["Solar", "solar", "Installed - Solar Power", "installed_solar_power"])
        biomass_col = _resolve_column(df, ["Biomass", "biomass", "Installed - Bio-Mass Power", "installed_bio_mass_power"])
        small_hydro_col = _resolve_column(df, ["Small Hydro", "small_hydro", "Installed - Small Hydro Power", "installed_small_hydro_power"])

        if not all([state_col, wind_col, solar_col, biomass_col, small_hydro_col]):
            return jsonify({"error": "Required columns missing in seasonal_df"}), 500

        # 1. Try exact match first
        sdf = df[df[state_col].astype(str).str.lower() == state_name.lower()].copy()
        
        # 2. SMART FALLBACK: If exact match fails, check for partial matches (like "Delhi" in "NCT of Delhi")
        if sdf.empty:
            for dataset_state in df[state_col].dropna().unique():
                ds_lower = str(dataset_state).lower().strip()
                req_lower = state_name.lower().strip()
                
                # Check if one is inside the other
                if (ds_lower in req_lower or req_lower in ds_lower) and len(ds_lower) > 3:
                    sdf = df[df[state_col].astype(str).str.lower() == ds_lower].copy()
                    break

        if sdf.empty:
            return jsonify({"error": "State not found"}), 404

        if month_col is None:
            year_col = _resolve_column(sdf, ["Year", "year"])
            if year_col is None:
                return jsonify({"error": "Required month/year columns missing in seasonal_df"}), 500
            month_col = "MONTH"
            month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            year_vals = pd.to_numeric(sdf[year_col], errors="coerce").fillna(2000).astype(int)
            sdf[month_col] = [month_order[(y - 2000) % 12] for y in year_vals]

        sdf["Total_Renewable"] = (
            pd.to_numeric(sdf[wind_col], errors="coerce")
            + pd.to_numeric(sdf[solar_col], errors="coerce")
            + pd.to_numeric(sdf[biomass_col], errors="coerce")
            + pd.to_numeric(sdf[small_hydro_col], errors="coerce")
        )

        grouped = (
            sdf.groupby(month_col, as_index=False)
            .agg(
                {
                    wind_col: "mean",
                    solar_col: "mean",
                    biomass_col: "mean",
                    small_hydro_col: "mean",
                    "Total_Renewable": "mean",
                }
            )
        )

        grouped = order_months(grouped, month_col=month_col)

        payload = [
            {
                "month": str(row[month_col]),
                "Solar": safe_float(row[solar_col]),
                "Wind": safe_float(row[wind_col]),
                "Biomass": safe_float(row[biomass_col]),
                "SmallHydro": safe_float(row[small_hydro_col]),
                "Total": safe_float(row["Total_Renewable"]),
            }
            for _, row in grouped.iterrows()
        ]
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.post("/predict")
def predict_output():
    try:
        model = current_app.config.get("rf_model")
        if model is None:
            return jsonify({"error": "rf_model not loaded"}), 500

        required_fields = [
            "ghi",
            "dni",
            "wind_speed_100m",
            "air_temp",
            "relative_humidity",
            "clearsky_ghi",
            "cloud_opacity",
            "precipitation_rate",
            "albedo",
        ]

        data = request.get_json(silent=True) or {}
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        input_df = pd.DataFrame([[data[field] for field in required_fields]], columns=required_fields)
        pred = model.predict(input_df)[0]

        return jsonify({"predicted_renewable": float(pred), "unit": "MW"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api_bp.get("/summary")
def get_summary():
    try:
        df = current_app.config.get("cluster_df")
        if df is None:
            return jsonify({"error": "cluster_df not loaded"}), 500

        state_col = _state_col(df)
        cluster_col = _cluster_col(df)
        total_col = _total_col(df)
        if not all([state_col, cluster_col, total_col]):
            return jsonify({"error": "Required columns missing in cluster_df"}), 500

        work = df.copy()
        work[total_col] = pd.to_numeric(work[total_col], errors="coerce")

        total_states = int(work[state_col].nunique())
        energy_hubs = int((work[cluster_col] == "Energy Hub").sum())
        energy_consumers = int((work[cluster_col] == "Energy Consumer").sum())

        hub_df = work[work[cluster_col] == "Energy Hub"].sort_values(total_col, ascending=False)
        if hub_df.empty:
            top_hub = None
            highest_output = None
        else:
            top_hub = str(hub_df.iloc[0][state_col])
            highest_output = safe_float(hub_df.iloc[0][total_col])

        return jsonify(
            {
                "total_states": total_states,
                "energy_hubs": energy_hubs,
                "energy_consumers": energy_consumers,
                "top_hub": top_hub,
                "highest_output_mw": highest_output,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
