"""
yield_forecast.py
─────────────────────────────────────────────────────────────────────────────
Hydroponic Lettuce Yield Forecasting System
Drop this file into your project, load your trained models, and call
`system.process_reading(features)` every time a new sensor reading arrives.

DEPENDENCIES:
    pip install pandas numpy xgboost joblib scikit-learn

QUICK START:
    1. Train and save models from your notebook:

        import joblib
        joblib.dump({'model': short_model, 'features': FEATURE_COLS},        'short_horizon_model.pkl')
        joblib.dump({'model': long_model,  'features': LONG_FEATURES_EXTRA}, 'long_horizon_model.pkl')

    2. Load and run here:

        from yield_forecast import load_system, make_reading
        system = load_system('short_horizon_model.pkl', 'long_horizon_model.pkl')

        reading = make_reading(
            growth_day=18, N_mgl=95, P_mgl=18, K_mgl=9.5,
            Ca_mgl=55, Mg_mgl=15, S_mgl=22, Fe_mgl=0.7,
            pH=6.1, temp_air=21.0, EC_estimated=1750,
        )
        result = system.process_reading(reading)
        print(result)
"""

import joblib
import pandas as pd
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

CYCLE_LENGTH = 32
HARVEST_DAYS = {14, 18, 21, 25, 28, 32}

# Nutrient thresholds for lettuce (mg/L)
THRESHOLDS = {
    'N_mgl':  {'warning': 80,  'critical': 40},
    'P_mgl':  {'warning': 8,   'critical': 4},
    'K_mgl':  {'warning': 8,   'critical': 4},
    'Ca_mgl': {'warning': 40,  'critical': 20},
    'Mg_mgl': {'warning': 10,  'critical': 5},
    'pH':     {'low_warning': 5.5, 'low_critical': 5.0,
               'high_warning': 6.8, 'high_critical': 7.2},
}


# ── Main system class ─────────────────────────────────────────────────────────

class YieldForecastSystem:
    """
    Feed sensor readings in chronologically and get yield forecasts + alerts.

    Parameters
    ----------
    short_model : XGBRegressor
        Predicts yield at the next harvest day (3–7 days out).
    long_model : XGBRegressor
        Predicts final yield at end of the grow cycle.
    short_features : list[str]
        Feature column names expected by short_model (in order).
    long_features : list[str]
        Feature column names expected by long_model (in order).
    alert_start_day : int
        Suppress alerts before this growth day (default 7 — early readings
        are too noisy to act on).
    """

    def __init__(self, short_model, long_model, short_features, long_features,
                 alert_start_day=7):
        self.short_model     = short_model
        self.long_model      = long_model
        self.short_features  = short_features
        self.long_features   = long_features
        self.alert_start_day = alert_start_day
        self.history         = []   # list of result dicts, one per reading

    # ── Public API ────────────────────────────────────────────────────────────

    def process_reading(self, features_dict: dict) -> dict:
        """
        Call this every time a new ISE sensor reading arrives.

        Parameters
        ----------
        features_dict : dict
            Keys are feature names (see make_reading() for a helper).
            Missing features are passed as NaN — XGBoost handles them.

        Returns
        -------
        dict with keys:
            growth_day        – int, day of this reading
            short_forecast_g  – float, predicted yield at next harvest (g)
            long_forecast_g   – float, predicted final cycle yield (g)
            alert             – str or None, human-readable alert message
            recommendation    – str or None, corrective action for farmer
            threshold_warnings– list[str], any nutrients below warning levels
        """
        short_row = self._make_row(features_dict, self.short_features)
        long_row  = self._make_row(features_dict, self.long_features)

        short_fc = float(self.short_model.predict(short_row)[0])
        long_fc  = float(self.long_model.predict(long_row)[0])

        current_day = int(features_dict.get('growth_day', 0))

        result = {
            'growth_day':        current_day,
            'short_forecast_g':  round(short_fc, 1),
            'long_forecast_g':   round(long_fc, 1),
            'features':          features_dict,
            'alert':             None,
            'recommendation':    None,
            'threshold_warnings': self._check_thresholds(features_dict),
        }

        # Only check for forecast drops after alert_start_day
        if self.history and current_day >= self.alert_start_day:
            prev       = self.history[-1]
            short_drop = prev['short_forecast_g'] - short_fc
            long_drop  = prev['long_forecast_g']  - long_fc
            short_pct  = short_drop / (prev['short_forecast_g'] + 1e-9) * 100
            long_pct   = long_drop  / (prev['long_forecast_g']  + 1e-9) * 100

            # Short-horizon alert: >15% or >5g drop
            if short_drop > max(5.0, 0.15 * prev['short_forecast_g']):
                culprit, change, direction = self._find_culprit(
                    prev['features'], features_dict)
                if self._is_harmful(culprit, direction):
                    result['alert'] = (
                        f"Short-horizon forecast dropped {short_drop:.1f}g "
                        f"({short_pct:.0f}%) since Day {prev['growth_day']}. "
                        f"Most likely cause: {culprit} changed {change}."
                    )
                    result['recommendation'] = self._get_recommendation(
                        culprit, features_dict)

            # Long-horizon alert: >20% or >10g drop
            if long_drop > max(10.0, 0.20 * prev['long_forecast_g']):
                long_alert = (
                    f" Long-horizon forecast also dropped {long_drop:.1f}g "
                    f"({long_pct:.0f}%) — review nutrient levels."
                )
                result['alert'] = (result['alert'] or '') + long_alert

        self.history.append(result)
        return result

    def compare_to_actual(self, growth_day: int, actual_fresh_g: float) -> dict | None:
        """
        Call this when the farmer weighs an actual harvest.
        Finds the most recent prediction made before this harvest day and
        compares it to reality.

        Returns dict with predicted_g, actual_g, error_g, error_pct,
        overestimated, significant (True if error > 20%).
        Returns None if no prior prediction exists.
        """
        for record in reversed(self.history):
            if record['growth_day'] is not None and record['growth_day'] < growth_day:
                predicted = record['short_forecast_g']
                diff      = predicted - actual_fresh_g
                pct       = diff / (predicted + 1e-9) * 100
                return {
                    'predicted_g':   round(predicted, 1),
                    'actual_g':      round(actual_fresh_g, 1),
                    'error_g':       round(diff, 1),
                    'error_pct':     round(pct, 1),
                    'overestimated': diff > 0,
                    'significant':   abs(pct) > 20,
                }
        return None

    def reset(self):
        """Clear history — call this at the start of each new grow cycle."""
        self.history = []

    def latest(self) -> dict | None:
        """Return the most recent result, or None if no readings yet."""
        return self.history[-1] if self.history else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_row(self, features_dict: dict, feature_list: list) -> pd.DataFrame:
        """Build a single-row DataFrame with correct dtypes for XGBoost."""
        row = pd.DataFrame([features_dict]).reindex(columns=feature_list)
        for col in row.columns:
            row[col] = pd.to_numeric(row[col], errors='coerce')
        return row

    def _find_culprit(self, prev_f: dict, curr_f: dict):
        """
        Find which watched feature changed most proportionally
        between two consecutive readings.
        Returns (feature_name, description_string, direction)
        """
        watch = ['N_mgl', 'P_mgl', 'K_mgl', 'Ca_mgl',
                 'pH', 'temp_air', 'EC_estimated']
        max_change, culprit, change_desc, direction = 0, 'unknown', '', 'unknown'

        for feat in watch:
            prev_val = prev_f.get(feat)
            curr_val = curr_f.get(feat)
            if prev_val is None or curr_val is None:
                continue
            try:
                pv, cv = float(prev_val), float(curr_val)
            except (TypeError, ValueError):
                continue
            pct = abs(cv - pv) / (abs(pv) + 1e-9)
            if pct > max_change:
                max_change  = pct
                culprit     = feat
                direction   = 'dropped' if cv < pv else 'increased'
                change_desc = f"from {pv:.1f} to {cv:.1f} ({direction})"

        return culprit, change_desc, direction

    def _is_harmful(self, culprit: str, direction: str) -> bool:
        """
        Return True only if the change is in a direction that would
        hurt yield — avoids false alerts when e.g. N increases.
        """
        harmful_map = {
            'N_mgl':        'dropped',
            'P_mgl':        'dropped',
            'K_mgl':        'dropped',
            'Ca_mgl':       'dropped',
            'EC_estimated': 'dropped',
            'temp_air':     'increased',
            'pH':           None,   # both directions can be bad for pH
        }
        if culprit not in harmful_map:
            return True
        expected = harmful_map[culprit]
        if expected is None:
            return True
        return direction == expected

    def _check_thresholds(self, features_dict: dict) -> list[str]:
        """Return list of human-readable warnings for nutrients below threshold."""
        warnings = []
        for ion, levels in THRESHOLDS.items():
            if ion == 'pH':
                val = features_dict.get('pH')
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if val < levels['low_critical']:
                    warnings.append(f"CRITICAL: pH {val:.1f} is dangerously low (< {levels['low_critical']})")
                elif val < levels['low_warning']:
                    warnings.append(f"WARNING: pH {val:.1f} is low (target 5.8–6.2)")
                elif val > levels['high_critical']:
                    warnings.append(f"CRITICAL: pH {val:.1f} is dangerously high (> {levels['high_critical']})")
                elif val > levels['high_warning']:
                    warnings.append(f"WARNING: pH {val:.1f} is high (target 5.8–6.2)")
            else:
                val = features_dict.get(ion)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                name = ion.replace('_mgl', '')
                if val <= levels['critical']:
                    warnings.append(
                        f"CRITICAL: [{name}] = {val:.1f} mg/L is below critical threshold "
                        f"({levels['critical']} mg/L)")
                elif val <= levels['warning']:
                    warnings.append(
                        f"WARNING: [{name}] = {val:.1f} mg/L is below warning threshold "
                        f"({levels['warning']} mg/L)")
        return warnings

    def _get_recommendation(self, culprit: str, features: dict) -> str:
        recs = {
            'N_mgl':
                'Nitrogen is depleting faster than expected. '
                'Add calcium nitrate to replenish N. Target [N] > 80 mg/L.',
            'P_mgl':
                'Phosphorus is dropping sharply. '
                'Check pH — P availability drops above pH 7.0. '
                'Target [P] > 8 mg/L. Add monopotassium phosphate if low.',
            'K_mgl':
                'Potassium is depleting. '
                'Check pH first — lower to 6.0–6.2 if above 6.5. '
                'Target [K] > 8 mg/L. Add potassium sulfate if deficient.',
            'Ca_mgl':
                'Calcium is dropping. Low Ca causes tip burn in lettuce. '
                'Ensure pH 6.0–6.5. Target [Ca] > 40 mg/L.',
            'pH':
                'pH has shifted outside optimal range. '
                'Target 5.8–6.2 for lettuce. '
                'Adjust with pH Up (KOH) or pH Down (H3PO4).',
            'temp_air':
                'Temperature is too high. Lettuce optimal: 18–22°C. '
                'High temps accelerate respiration and reduce yield.',
            'EC_estimated':
                'Total ionic concentration dropped sharply. '
                'Test individual ions and compare to targets. '
                'Full nutrient solution replacement may be needed.',
        }
        return recs.get(
            culprit,
            'Review all nutrient and environmental readings against target values.')


# ── Loader function ───────────────────────────────────────────────────────────

def load_system(short_pkl: str, long_pkl: str,
                alert_start_day: int = 7) -> YieldForecastSystem:
    """
    Load saved model .pkl files and return a ready-to-use YieldForecastSystem.

    Parameters
    ----------
    short_pkl : str
        Path to short_horizon_model.pkl saved with joblib.
    long_pkl : str
        Path to long_horizon_model.pkl saved with joblib.
    alert_start_day : int
        Don't fire alerts before this growth day (default 7).

    Example
    -------
    system = load_system('short_horizon_model.pkl', 'long_horizon_model.pkl')
    """
    short_pkg = joblib.load(short_pkl)
    long_pkg  = joblib.load(long_pkl)

    return YieldForecastSystem(
        short_model     = short_pkg['model'],
        long_model      = long_pkg['model'],
        short_features  = short_pkg['features'],
        long_features   = long_pkg['features'],
        alert_start_day = alert_start_day,
    )


# ── Helper to build a reading dict ───────────────────────────────────────────

def make_reading(
    growth_day: int,
    N_mgl:      float = None,
    P_mgl:      float = None,
    K_mgl:      float = None,
    Ca_mgl:     float = None,
    Mg_mgl:     float = None,
    S_mgl:      float = None,
    Fe_mgl:     float = None,
    pH:         float = None,
    temp_air:   float = None,
    EC_estimated: float = None,
    # Depletion rates — pass if you have them, else leave None
    N_depletion_rate:  float = None,
    P_depletion_rate:  float = None,
    K_depletion_rate:  float = None,
    Ca_depletion_rate: float = None,
    Mg_depletion_rate: float = None,
    S_depletion_rate:  float = None,
    # Fraction remaining — pass if you have them
    N_fraction_remaining:  float = None,
    P_fraction_remaining:  float = None,
    K_fraction_remaining:  float = None,
    Ca_fraction_remaining: float = None,
    Mg_fraction_remaining: float = None,
    S_fraction_remaining:  float = None,
    # Treatment concentrations (from meta file — leave None for live use)
    trt_N_mgl: float = None,
    trt_P_mgl: float = None,
    trt_K_mgl: float = None,
) -> dict:
    """
    Convenience function to build a features dict for process_reading().

    Computed features (ratios, EC, growth stage) are auto-calculated if the
    underlying values are provided. Depletion rates and fraction remaining
    require historical context — if you don't have them the model will treat
    them as NaN, which XGBoost handles gracefully.

    Example
    -------
    reading = make_reading(
        growth_day=18, N_mgl=95, P_mgl=18, K_mgl=9.5,
        Ca_mgl=55, Mg_mgl=15, S_mgl=22, Fe_mgl=0.7,
        pH=6.1, temp_air=21.0,
    )
    """
    # Auto-compute EC if not provided but ions are available
    if EC_estimated is None:
        try:
            EC_estimated = (
                (N_mgl  or 0) / 14.0  * 7.34 +
                (K_mgl  or 0) / 39.1  * 7.35 +
                (Ca_mgl or 0) / 20.0  * 11.9 +
                (Mg_mgl or 0) / 12.2  * 10.6 +
                (P_mgl  or 0) / 31.0  * 3.69 +
                (S_mgl  or 0) / 16.0  * 8.0
            )
        except TypeError:
            EC_estimated = None

    # Auto-compute ratios
    try:
        N_to_K_ratio  = N_mgl / K_mgl  if N_mgl and K_mgl  else None
    except (TypeError, ZeroDivisionError):
        N_to_K_ratio = None
    try:
        N_to_Ca_ratio = N_mgl / Ca_mgl if N_mgl and Ca_mgl else None
    except (TypeError, ZeroDivisionError):
        N_to_Ca_ratio = None
    try:
        Ca_to_Mg_ratio = Ca_mgl / Mg_mgl if Ca_mgl and Mg_mgl else None
    except (TypeError, ZeroDivisionError):
        Ca_to_Mg_ratio = None

    return {
        # Raw concentrations
        'N_mgl':   N_mgl,
        'P_mgl':   P_mgl,
        'K_mgl':   K_mgl,
        'Ca_mgl':  Ca_mgl,
        'Mg_mgl':  Mg_mgl,
        'S_mgl':   S_mgl,
        'Fe_mgl':  Fe_mgl,
        # Depletion rates
        'N_depletion_rate':  N_depletion_rate,
        'P_depletion_rate':  P_depletion_rate,
        'K_depletion_rate':  K_depletion_rate,
        'Ca_depletion_rate': Ca_depletion_rate,
        'Mg_depletion_rate': Mg_depletion_rate,
        'S_depletion_rate':  S_depletion_rate,
        # Fraction remaining
        'N_fraction_remaining':  N_fraction_remaining,
        'P_fraction_remaining':  P_fraction_remaining,
        'K_fraction_remaining':  K_fraction_remaining,
        'Ca_fraction_remaining': Ca_fraction_remaining,
        'Mg_fraction_remaining': Mg_fraction_remaining,
        'S_fraction_remaining':  S_fraction_remaining,
        # Ratios (auto-computed)
        'N_to_K_ratio':   N_to_K_ratio,
        'N_to_Ca_ratio':  N_to_Ca_ratio,
        'Ca_to_Mg_ratio': Ca_to_Mg_ratio,
        # Environmental
        'EC_estimated': EC_estimated,
        'temp_air':     temp_air,
        'pH':           pH,
        # Growth stage (auto-computed)
        'growth_day':               growth_day,
        'normalized_growth_stage':  growth_day / CYCLE_LENGTH,
        'is_harvest_day':           int(growth_day in HARVEST_DAYS),
        # Treatment concentrations (leave None for live deployment)
        'trt_N_mgl': trt_N_mgl,
        'trt_P_mgl': trt_P_mgl,
        'trt_K_mgl': trt_K_mgl,
    }


# ── Save helper (call from your notebook) ────────────────────────────────────

def save_models(short_model, short_features: list,
                long_model,  long_features: list,
                short_path: str = 'short_horizon_model.pkl',
                long_path:  str = 'long_horizon_model.pkl'):
    """
    Save both models from your notebook so they can be loaded here.

    Call this at the end of your training notebook:
        from yield_forecast import save_models
        save_models(short_model, FEATURE_COLS, long_model, LONG_FEATURES_EXTRA)
    """
    joblib.dump({'model': short_model, 'features': short_features}, short_path)
    joblib.dump({'model': long_model,  'features': long_features},  long_path)
    print(f"Saved short-horizon model → {short_path}")
    print(f"Saved long-horizon model  → {long_path}")


# ── Example usage (runs when you execute this file directly) ──────────────────

if __name__ == '__main__':
    print("yield_forecast.py — usage example")
    print("=" * 60)
    print("""
To use this module:

1. At the end of your training notebook, save the models:

    import joblib
    joblib.dump({'model': short_model, 'features': FEATURE_COLS},
                'short_horizon_model.pkl')
    joblib.dump({'model': long_model,  'features': LONG_FEATURES_EXTRA},
                'long_horizon_model.pkl')

2. In your application:

    from yield_forecast import load_system, make_reading

    system = load_system('short_horizon_model.pkl', 'long_horizon_model.pkl')

    # Each time a sensor reading comes in:
    reading = make_reading(
        growth_day=18,
        N_mgl=95,   P_mgl=18,  K_mgl=9.5,
        Ca_mgl=55,  Mg_mgl=15, S_mgl=22,  Fe_mgl=0.7,
        pH=6.1,     temp_air=21.0,
    )
    result = system.process_reading(reading)

    print(f"Short forecast: {result['short_forecast_g']}g")
    print(f"Long forecast:  {result['long_forecast_g']}g")

    if result['alert']:
        print(f"ALERT:  {result['alert']}")
        print(f"ACTION: {result['recommendation']}")

    for w in result['threshold_warnings']:
        print(f"THRESHOLD: {w}")

    # When farmer enters actual harvest weight:
    val = system.compare_to_actual(growth_day=18, actual_fresh_g=12.9)
    if val and val['significant']:
        print(f"Model was off by {val['error_pct']}% — consider retraining.")

    # Start of a new grow cycle:
    system.reset()
""")