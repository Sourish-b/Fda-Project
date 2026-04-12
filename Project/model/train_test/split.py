import os

import pandas as pd
from sklearn.model_selection import train_test_split


def _resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_splits(df):
    """Create or load train/test splits for renewable output modeling."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    x_train_path = os.path.join(results_dir, "X_train.csv")
    x_test_path = os.path.join(results_dir, "X_test.csv")
    y_train_path = os.path.join(results_dir, "y_train.csv")
    y_test_path = os.path.join(results_dir, "y_test.csv")

    if all(os.path.exists(path) for path in [x_train_path, x_test_path, y_train_path, y_test_path]):
        print("Splits loaded from results/ — not re-splitting.")
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        if y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]

        return X_train, X_test, y_train, y_test

    data = df.copy()

    wind_col = _resolve_column(data, ["Wind", "wind"])
    solar_col = _resolve_column(data, ["Solar", "solar"])
    biomass_col = _resolve_column(data, ["Biomass", "biomass"])
    small_hydro_col = _resolve_column(data, ["Small Hydro", "small_hydro"])
    total_col = _resolve_column(data, ["Total_Renewable", "total_renewable"])

    if total_col is None:
        if not all([wind_col, solar_col, biomass_col, small_hydro_col]):
            raise KeyError("Missing columns for Total_Renewable creation: Wind, Solar, Biomass, Small Hydro")
        data["Total_Renewable"] = (
            pd.to_numeric(data[wind_col], errors="coerce")
            + pd.to_numeric(data[solar_col], errors="coerce")
            + pd.to_numeric(data[biomass_col], errors="coerce")
            + pd.to_numeric(data[small_hydro_col], errors="coerce")
        )
        total_col = "Total_Renewable"

    feature_map = {
        "ghi": ["ghi", "GHI"],
        "dni": ["dni", "DNI"],
        "wind_speed_100m": ["wind_speed_100m", "Wind Speed 100m", "wind_speed_at_100m"],
        "air_temp": ["air_temp", "Air Temp", "air_temperature"],
        "relative_humidity": ["relative_humidity", "Relative Humidity"],
        "clearsky_ghi": ["clearsky_ghi", "Clearsky GHI", "clear_sky_ghi"],
        "cloud_opacity": ["cloud_opacity", "Cloud Opacity"],
        "precipitation_rate": ["precipitation_rate", "Precipitation Rate"],
        "albedo": ["albedo", "Albedo"],
    }

    missing_features = []
    resolved_features = {}
    for canonical_name, candidates in feature_map.items():
        found_col = _resolve_column(data, candidates)
        if found_col is None:
            missing_features.append(canonical_name)
        else:
            resolved_features[canonical_name] = found_col

    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    X = data[[resolved_features[name] for name in feature_map.keys()]].copy()
    X.columns = list(feature_map.keys())
    y = pd.to_numeric(data[total_col], errors="coerce")

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_frame(name="Total_Renewable").to_csv(y_train_path, index=False)
    y_test.to_frame(name="Total_Renewable").to_csv(y_test_path, index=False)

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    print("Feature list:", list(feature_map.keys()))

    return X_train, X_test, y_train, y_test
