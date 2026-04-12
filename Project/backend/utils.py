import pandas as pd


def normalize_state_name(name):
    """Strip whitespace and convert a state name to title case."""
    if name is None:
        return ""
    return str(name).strip().title()


def find_state(df, state_name):
    """Case-insensitive state lookup in Name of State/UT column."""
    if df is None:
        return None

    state_col = None
    for col in ["Name of State/UT", "name_of_state_ut", "State", "state"]:
        if col in df.columns:
            state_col = col
            break

    if state_col is None:
        return None

    normalized = normalize_state_name(state_name).lower()
    matched = df[df[state_col].astype(str).str.strip().str.lower() == normalized]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def order_months(df, month_col="MONTH"):
    """Return a DataFrame sorted in calendar month order."""
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    out = df.copy()
    out[month_col] = out[month_col].astype(str).str.strip().str.upper().str[:3]
    out[month_col] = pd.Categorical(out[month_col], categories=month_order, ordered=True)
    out = out.sort_values(month_col).reset_index(drop=True)
    return out


def safe_float(value):
    """Convert value to rounded float, returning 0.0 for invalid values."""
    try:
        val = float(value)
        if pd.isna(val):
            return 0.0
        return round(val, 2)
    except (TypeError, ValueError):
        return 0.0
