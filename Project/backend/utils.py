import pandas as pd


# Add this dictionary at the top of the file, or right above the function
STATE_ALIASES = {
    "orissa": "odisha",
    "delhi": "nct of delhi",
    "new delhi": "nct of delhi",
    "jammu": "jammu and kashmir",
    "jammu & kashmir": "jammu and kashmir",
    "j&k": "jammu and kashmir",
    "jk": "jammu and kashmir",
    "uttaranchal": "uttarakhand",
    "uttarkhand": "uttarakhand"
}

def normalize_state_name(name):
    """Strip whitespace, resolve common aliases, and convert to title case."""
    if name is None:
        return ""
    
    # Clean the incoming name
    clean_name = str(name).strip().lower()
    
    # Check if the name needs to be translated using our alias dictionary
    if clean_name in STATE_ALIASES:
        clean_name = STATE_ALIASES[clean_name]
        
    return clean_name.title()

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
    
    # Clean the dataset column for comparison
    clean_col = df[state_col].astype(str).str.strip().str.lower()
    matched = df[clean_col == normalized]
    
    # DEBUG TRACKER: If it fails to match, print out exactly what is wrong
    if matched.empty:
        print("\n" + "="*40)
        print(f"❌ MISMATCH DETECTED")
        print(f"1. You clicked on Map: '{state_name}'")
        print(f"2. Backend translated to: '{normalized}'")
        print(f"3. BUT your Dataset ONLY contains these states:\n{df[state_col].unique().tolist()}")
        print("="*40 + "\n")
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
