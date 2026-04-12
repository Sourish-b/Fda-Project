import re
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


STATE_NAME_MAP = {
    "Andaman & Nicobar": "Andaman and Nicobar Islands",
    "Delhi": "NCT of Delhi",
    "J&K": "Jammu & Kashmir",
}


def _to_snake_case(col_name):
    col = col_name.strip().lower()
    col = re.sub(r"[%/()-]", " ", col)
    col = re.sub(r"[^a-z0-9\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col.strip("_")


def clean_data(df):
    """Clean and standardize renewable-energy dataframe fields for downstream analysis."""
    df = df.copy()

    if "Name of State/UT" in df.columns:
        df["Name of State/UT"] = df["Name of State/UT"].astype(str).str.strip()
        df["Name of State/UT"] = df["Name of State/UT"].replace(STATE_NAME_MAP)

    normalized_columns = {_to_snake_case(col): col for col in df.columns}

    explicit_rename = {
        "wind_speed_100m": "wind_speed_100m",
        "wind_speed_at_100m": "wind_speed_100m",
        "air_temp": "air_temp",
        "air_temperature": "air_temp",
        "relative_humidity": "relative_humidity",
    }

    rename_map = {}
    for normalized, original in normalized_columns.items():
        rename_map[original] = explicit_rename.get(normalized, normalized)

    df = df.rename(columns=rename_map)

    required_energy_cols = ["wind", "solar", "biomass", "small_hydro"]

    energy_aliases = {
        "wind": ["wind", "wind_mw", "wind_power", "wind_power_mw"],
        "solar": ["solar", "solar_mw", "solar_power", "solar_power_mw"],
        "biomass": ["biomass", "biomass_mw", "biomass_power", "biomass_power_mw"],
        "small_hydro": [
            "small_hydro",
            "small_hydro_mw",
            "small_hydropower",
            "small_hydropower_mw",
        ],
    }

    for target_col, candidates in energy_aliases.items():
        if target_col not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[target_col] = df[candidate]
                    break
            if target_col not in df.columns:
                df[target_col] = pd.NA

    df = df.dropna(subset=required_energy_cols, how="all")
    df = df.reset_index(drop=True)

    if "Name of State/UT" in df.columns:
        print(df["Name of State/UT"].unique())

    return df


def _resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def engineer_features(df):
    """Create total renewable feature and state-level mean profile."""
    df = df.copy()

    wind_col = _resolve_column(df, ["Wind", "wind"])
    solar_col = _resolve_column(df, ["Solar", "solar"])
    biomass_col = _resolve_column(df, ["Biomass", "biomass"])
    small_hydro_col = _resolve_column(df, ["Small Hydro", "small_hydro"])

    if not all([wind_col, solar_col, biomass_col, small_hydro_col]):
        raise KeyError("Required renewable columns not found: Wind, Solar, Biomass, Small Hydro")

    df["Total_Renewable"] = (
        pd.to_numeric(df[wind_col], errors="coerce")
        + pd.to_numeric(df[solar_col], errors="coerce")
        + pd.to_numeric(df[biomass_col], errors="coerce")
        + pd.to_numeric(df[small_hydro_col], errors="coerce")
    )

    state_col = _resolve_column(df, ["Name of State/UT", "name_of_state_ut"])
    if not state_col:
        raise KeyError("State column not found: Name of State/UT")

    grouped_df = df.groupby(state_col, as_index=False).mean(numeric_only=True)
    if state_col != "Name of State/UT":
        grouped_df = grouped_df.rename(columns={state_col: "Name of State/UT"})

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "state_profiles.csv")
    grouped_df.to_csv(output_path, index=False)

    return grouped_df


def scale_features(df, feature_cols):
    """Scale selected feature columns and persist the fitted scaler."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_cols])

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    return scaled_array, scaler
