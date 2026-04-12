import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _energy_column_map(df):
    wind_col = _resolve_column(df, ["Wind", "wind"])
    solar_col = _resolve_column(df, ["Solar", "solar"])
    biomass_col = _resolve_column(df, ["Biomass", "biomass"])
    small_hydro_col = _resolve_column(df, ["Small Hydro", "small_hydro"])

    if not all([wind_col, solar_col, biomass_col, small_hydro_col]):
        raise KeyError(
            "Required columns missing. Needed: Wind, Solar, Biomass, Small Hydro"
        )

    return {
        "Wind": wind_col,
        "Solar": solar_col,
        "Biomass": biomass_col,
        "Small Hydro": small_hydro_col,
    }


def _month_categorical(series):
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    cleaned = series.astype(str).str.strip().str.upper().str[:3]
    return pd.Categorical(cleaned, categories=month_order, ordered=True), month_order


def plot_monthly_heatmap(df):
    """Plot and save a monthly state-wise renewable output heatmap."""
    data = df.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    month_col = _resolve_column(data, ["MONTH", "month"])
    energy_cols = _energy_column_map(data)

    if not all([state_col, month_col]):
        raise KeyError(
            "Required columns missing. Needed: Name of State/UT, MONTH"
        )

    data["Total_Renewable"] = (
        pd.to_numeric(data[energy_cols["Wind"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Solar"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Biomass"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Small Hydro"]], errors="coerce")
    )

    data[month_col], month_order = _month_categorical(data[month_col])

    grouped = (
        data.groupby([state_col, month_col], as_index=False)["Total_Renewable"]
        .mean()
    )

    heatmap_df = grouped.pivot(index=state_col, columns=month_col, values="Total_Renewable")
    heatmap_df = heatmap_df.reindex(columns=month_order)

    plt.figure(figsize=(16, 12))
    sns.heatmap(heatmap_df, cmap="YlOrRd", annot=False)
    plt.title("Monthly Renewable Output by State (MW)")
    plt.xlabel("MONTH")
    plt.ylabel("Name of State/UT")
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "monthly_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_states_trend(df):
    """Plot and save monthly trend lines for top 5 states by mean total renewable output."""
    data = df.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    month_col = _resolve_column(data, ["MONTH", "month"])
    energy_cols = _energy_column_map(data)

    if not all([state_col, month_col]):
        raise KeyError(
            "Required columns missing. Needed: Name of State/UT, MONTH"
        )

    data["Total_Renewable"] = (
        pd.to_numeric(data[energy_cols["Wind"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Solar"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Biomass"]], errors="coerce")
        + pd.to_numeric(data[energy_cols["Small Hydro"]], errors="coerce")
    )

    data[month_col], month_order = _month_categorical(data[month_col])

    top_states = (
        data.groupby(state_col, as_index=False)["Total_Renewable"]
        .mean()
        .sort_values("Total_Renewable", ascending=False)
        .head(5)[state_col]
        .tolist()
    )

    trend_df = (
        data[data[state_col].isin(top_states)]
        .groupby([state_col, month_col], as_index=False)["Total_Renewable"]
        .mean()
    )

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=trend_df,
        x=month_col,
        y="Total_Renewable",
        hue=state_col,
        style=state_col,
        markers=True,
        dashes=False,
        linewidth=2,
        palette="tab10",
    )
    plt.title("Top 5 States Monthly Renewable Trend (MW)")
    plt.xlabel("MONTH")
    plt.ylabel("MW Output")
    plt.legend(title="Name of State/UT", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "top5_monthly_trend.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    for state_name in top_states[:2]:
        plot_energy_mix(data, state_name)


def plot_energy_mix(df, state_name):
    """Plot and save monthly stacked energy mix for a given state."""
    data = df.copy()

    state_col = _resolve_column(data, ["Name of State/UT", "name_of_state_ut"])
    month_col = _resolve_column(data, ["MONTH", "month"])
    energy_cols = _energy_column_map(data)

    if not all([state_col, month_col]):
        raise KeyError("Required columns missing. Needed: Name of State/UT, MONTH")

    state_df = data[data[state_col] == state_name].copy()
    if state_df.empty:
        raise ValueError(f"No data found for state: {state_name}")

    state_df[month_col], month_order = _month_categorical(state_df[month_col])

    grouped = (
        state_df.groupby(month_col, as_index=False)[list(energy_cols.values())]
        .mean()
        .rename(columns={
            energy_cols["Wind"]: "Wind",
            energy_cols["Solar"]: "Solar",
            energy_cols["Biomass"]: "Biomass",
            energy_cols["Small Hydro"]: "Small Hydro",
        })
    )

    grouped = grouped.set_index(month_col).reindex(month_order)
    grouped = grouped[["Wind", "Solar", "Biomass", "Small Hydro"]]

    plt.figure(figsize=(14, 8))
    grouped.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
    plt.title(f"Monthly Energy Mix - {state_name}")
    plt.xlabel("MONTH")
    plt.ylabel("MW Output")
    plt.legend(title="Energy Type", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    safe_state = str(state_name).replace("/", "_").replace("\\", "_").strip()
    output_path = os.path.join(save_dir, f"energy_mix_{safe_state}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
