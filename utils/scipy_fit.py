import numpy as np
import json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# TODO: pythonize this (make state live where it should, properly handle all the params, generify where it should be)

param_name_to_index = {
        'F12': 0, 'F21': 1, 'K12': 2, 'K21': 3, 'K34': 4, 'K43': 5,
        'K56': 6, 'K65': 7, 'K78': 8, 'K87': 9, 'K910': 10,
        'f12': 11, 'f21': 12, 'k21': 13, 'k34': 14, 'k43': 15,
        'k56': 16, 'k65': 17, 'k78': 18, 'k87': 19, 'k910': 20,
        'knfb': 21, 's12': 22, 's21': 23, 'totERK': 24, 'totKTR': 25,
        'totMEK': 26, 'totNFB': 27, 'totRAF': 28, 'totRAS': 29
    }

def load_default_params():
    with open("egfr_fit_transient_1_params.json", "r") as f:
        params_dict = json.load(f)
        adv_obj = params_dict.get('params')
        if adv_obj is not None:
          params_dict = adv_obj
    return np.array(list(params_dict.values()))

# EGFR model
def egfr_model(t, y, params):
    """EGFR model adapted for SciPy"""
    RAS_s, RAF_s, MEK_s, ERK_s, NFB_s, KTR_s = y

    # Light function parameters (passed at end of params array)
    light_start_ms, light_duration_ms, light_intensity = params[-3:]

    # Light function
    t_ms = t * 1000.0
    if light_start_ms <= t_ms <= light_start_ms + light_duration_ms:
        light = light_intensity
    else:
        light = 0.0

    # This is a simplified subset - expand with your full parameter set
    F12, F21, K12, K21, K34, K43, K56, K65, K78, K87, K910 = params[0:11]
    f12, f21, k21, k34, k43, k56, k65, k78, k87, k910, knfb = params[11:22]
    s12, s21, totERK, totKTR, totMEK, totNFB, totRAF, totRAS = params[22:30]

    # Conservation laws
    RAS = totRAS - RAS_s
    RAF = totRAF - RAF_s
    MEK = totMEK - MEK_s
    ERK = totERK - ERK_s
    NFB = totNFB - NFB_s
    KTR = totKTR - KTR_s

    # Differential equations 
    dRAS_s = light * (RAS/(K12 + RAS)) - k21 * (RAS_s/(K21 + RAS_s))
    dRAF_s = k34 * RAS_s * (RAF/(K34 + RAF)) - (knfb * NFB_s + k43) * (RAF_s/(K43 + RAF_s))
    dMEK_s = k56 * RAF_s * (MEK/(K56 + MEK)) - k65 * (MEK_s/(K65 + MEK_s))
    dERK_s = k78 * MEK_s * (ERK/(K78 + ERK)) - k87 * (ERK_s/(K87 + ERK_s))
    dNFB_s = f12 * ERK_s * (NFB/(F12 + NFB)) - f21 * (NFB_s/(F21 + NFB_s))
    dKTR_s = (k910 * ERK_s * (KTR/(K910 + KTR)) + s12 * KTR) - s21 * KTR_s

    return [dRAS_s, dRAF_s, dMEK_s, dERK_s, dNFB_s, dKTR_s]

def prepare_dataframe(df):
    """Convert your DataFrame structure to standard format"""

    # Convert frame to time in seconds
    # Frame 1 = 0 seconds, Frame 2 = 60 seconds, etc.
    df = df.copy()
    df['time'] = (df['frame'] - 1) * 60.0  # Convert frame to seconds

    # Rename columns to standard names
    df['pulse_duration_ms'] = df['stim_exposure']  # Stimulation duration
    df['KTR_s'] = df['cnr_norm']  # Normalized KTR presence

    # Calculate start time (all stimulation begins at frame 10)
    df['start_time_ms'] = (10 - 1) * 60 * 1000  # Frame 10 = 540 seconds = 540000 ms

    return df

def estimate_parameters(df, params_to_fit, fixed_params=None):
    """
    Main parameter estimation function
    """

    if fixed_params is None:
        fixed_params = {}

    # Load default parameters
    try:
        default_params = load_default_params()
    except:
        # Fallback if JSON file not available
        default_params = np.ones(30)
        print("Warning: Could not load default parameters, using ones(30)")

    # Prepare DataFrame
    df_processed = prepare_dataframe(df)

    # Parameter name to index mapping (adapt to your Julia parameter order)

    param_names = list(params_to_fit.keys())
    param_indices = [param_name_to_index[name] for name in param_names]

    y0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.1]  # Initial conditions

    def objective(p_fit_values):
        # Build full parameter vector
        p_full = default_params.copy()

        # Insert fitted parameters
        for i, param_idx in enumerate(param_indices):
            p_full[param_idx] = p_fit_values[i]

        # Insert fixed parameters
        for param_name, value in fixed_params.items():
            if param_name in param_name_to_index:
                idx = param_name_to_index[param_name]
                p_full[idx] = value

        total_loss = 0.0

        # Group by unique experiments (stim_exposure, uid combinations)
        unique_experiments = df_processed[['pulse_duration_ms', 'uid']].drop_duplicates()

        for _, exp_row in unique_experiments.iterrows():
            exp_data = df_processed[
                (df_processed['pulse_duration_ms'] == exp_row['pulse_duration_ms']) & 
                (df_processed['uid'] == exp_row['uid'])
            ].sort_values('time')

            times = exp_data['time'].values
            ktr_data = exp_data['KTR_s'].values

            # Light parameters: [start_time_ms, duration_ms, intensity]
            start_time_ms = exp_data['start_time_ms'].iloc[0]
            pulse_duration_ms = exp_row['pulse_duration_ms']
            light_params = [start_time_ms, pulse_duration_ms, 2.0]

            # Add light parameters to full parameter vector
            params_with_light = np.concatenate([p_full, light_params])

            try:
                sol = solve_ivp(
                    lambda t, y: egfr_model(t, y, params_with_light),
                    [times[0], times[-1]], y0, 
                    t_eval=times, method='LSODA', rtol=1e-8
                )

                if sol.success:
                    ktr_pred = sol.y[5]  # KTR_s is 6th state (index 5)
                    loss = np.sum((ktr_pred - ktr_data)**2)
                    total_loss += loss
                else:
                    return 1e10

            except Exception as e:
                print(f"Error solving ODE: {e}")
                return 1e10

        return total_loss

    # Initial parameter guess
    p0 = [params_to_fit[name] for name in param_names]

    # Optimization
    result = minimize(
        objective, p0, 
        method='L-BFGS-B',
        bounds=[(0.001, 5)] * len(p0),
        options={'maxiter': 1000}
    )

    # Package results
    fitted_params = dict(zip(param_names, result.x))

    return {
        'fitted_params': fitted_params,
        'loss': result.fun,
        'success': result.success,
        'message': result.message,
        'n_experiments': len(df_processed[['pulse_duration_ms', 'uid']].drop_duplicates())
    }

# Usage with your actual DataFrame structure
def run_parameter_estimation(df, experiment_name=None):

    assert experiment_name is not None


    # Define parameters to fix (optional)
    fixed_params = {
        "totERK":1,
        "totKTR":1,
        "totMEK":1,
        "totNFB":1,
        "totRAF":1,
        "totRAS":1,
    }
    defp = load_default_params()
    par_idx = param_name_to_index
    params_to_fit = {
        k:defp[par_idx[k]] for k in par_idx.keys() if k not in fixed_params.keys()
    }

    print(f"DataFrame shape: {df.shape}")
    print(f"Unique cells (uid): {df['uid'].nunique()}")
    print(f"Unique stimulation exposures: {sorted(df['stim_exposure'].unique())}")
    print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")

    # Run estimation
    results = estimate_parameters(df, params_to_fit, fixed_params)
    results['fixed'] = fixed_params

    print("\n=== Results ===")
    print(f"Optimization successful: {results['success']}")
    print(f"Final loss: {results['loss']:.6f}")
    print(f"Number of experiments: {results['n_experiments']}")
    print(f"Fitted parameters:")
    tosave = {"params":results['fitted_params'], "meta":results}
    for fp,fpv in fixed_params.items(): tosave['params'][fp] = fpv
    with open(f'{experiment_name}.json', 'x') as f: json.dump(tosave, f)
    print('Saved fitted params to', f'{experiment_name}.json')

    for name, value in results['fitted_params'].items():
        print(f"  {name}: {value:.4f}")

    return results

# Example usage:
# results = run_parameter_estimation(your_dataframe)

class Model:
    params = []
    eqs = []

