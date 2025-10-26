import numpy as np
from datetime import datetime
from sympy import Symbol, Function, Eq, Derivative, lambdify, Expr
from sympy.core.relational import Equality
from sympy.abc import t
from typing import List, Dict, TypedDict, NotRequired, AnyStr
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from pathlib import Path
from utils.utils import RESULTS_PATH
import json

class EquationDescription(TypedDict):
    equations:List[Equality]
    constraints: NotRequired[List[Equality]]
    symbols:NotRequired[List[Symbol]]
    base_equations: NotRequired[List[Equality]]

'''
TODO:   Think about time dilation, as a lot of the signalling phenomena happen on timescale 
        invisible from frame point of view (50ms pulse of light looks identical to every other if
        our time resolution is 1 frame (1 second)). 
        On the second thought, that should probably happen outside of this function,
        maybe on the level of preparing the dataframe? Just a helper func or sth?
'''

class Model:
    name = ''
    params = []    
    eqs = []
    fit_result = None
    def __init__(self, name, states, parameters, model_definition, t_func, t_dep, ivp_method='LSODA', minimizer_method='L-BFGS-B', group_to_light=None):
        '''
        We give it a name, we give it all the symbolic equations, and params (defaults?)

        Pipeline:
            data ---------------------------------------.
                                                        |
            eqs + params + light_fn -> numeric system  -L--> make loss function -> feed to minimizer

        What's a good API to insert EQs? Do i need to input symbol for t?           
        '''
        assert (single_tf := t_func is not None) != (multi_tf := group_to_light is not None) and (single_tf or multi_tf), "Either single time function or multi-time function"
        self.name = name
        self.group_to_light = group_to_light
        self.states = states
        self.parameters = parameters
        self.model_definition_f = model_definition
        model = model_definition(parameters, states)
        self.eqs = model['equations']
        self.symbols = model['symbols']
        self.base_equations = model['base_equations']
        assert type(t_dep) == str and t_dep in self.symbols.keys(), 'time-dependant variable must exist within equations'
        self.ivp_method = ivp_method
        self.minimizer_method = minimizer_method
        self.t_func = t_func # function, f.ex def light_fn(t, args*)
        self.t_dep = t_dep
                
    def make_numerical(self, module='numpy'):

        free_vars = set()
        for eq in self.eqs: free_vars.update(eq.rhs.free_symbols)
        # Find which state names have their corresponding symbols in free_vars
        active_states = []
        for state_name in self.states:
            if self.symbols[state_name] in free_vars:
                active_states.append(state_name)
        # ^ cannot just use self.states from the start cause the 
        #   self.transform() could have reduced number of parameters.
        
        self.active_states = active_states
        states = [self.symbols[s] for s in active_states]

        parameters = [self.symbols[p] for p in self.parameters]

        arg_list = (t, *states, *parameters, self.symbols[self.t_dep])
        numerical_funcs = [lambdify(arg_list, eq.rhs, modules=module )
                           for eq in self.eqs ]

        def system(t, y, params, t_func, t_args=None):
            res = t_func(t, t_args)
            # This here is hella expensive. Get rid of this.
            #y = np.asarray(y, dtype=np.float32)
            args = [t, *y, *params, res]
            return [f(*args) for f in numerical_funcs]
            
        return system

    def fit(self, dataframe, y0, parameters, t_args=None):
        y0 =  np.asarray(y0, dtype=np.float64)
        """
        Fit model parameters to experimental data.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            Must contain columns:
            - 'time': float, time points in seconds
            - 'y': float, observed values (e.g., KTR levels)
            - 'group': str/int, experiment/cell identifier for grouping
        y0 : array_like
            Initial conditions for the ODE system
        parameters : dict or str or PosixPath
            If dict: {param_name: initial_value} for parameters to optimize
            If str or PosixPath: path to JSON config file with parameter defaults
        t_args : dict
            Arguments to pass to the time-dependent function (e.g., light intensity)
        """
        # Handle parameters input - could be dict, str, or PosixPath
        config_path = None
        if isinstance(parameters, (str, Path)):
            config_path = Path(parameters)
            parameters = self.read_config(config_path)
        elif not isinstance(parameters, dict):
            raise ValueError("parameters must be dict, str, or PosixPath")

        if t_args is None:
            t_args = {}
        else:
            assert type(t_args) == dict, "t_args need to be a dictionary \
                (so i can automatically put generic keys in there, like a group_id)"

        base_t_args = dict(t_args)
        # Validate input dataframe
        required_cols = {'time', 'y', 'group'}
        if not required_cols.issubset(dataframe.columns):
            missing = required_cols - set(dataframe.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.parameters_to_fit = parameters
        param_names = list(parameters.keys())
        
        # Get the numerical system
        system = self.make_numerical()

        def objective_log(p_fit_log_values):
            # Transform from log space back to normal space
            p_fit_values = np.exp(p_fit_log_values)
            return objective_normal(p_fit_values)

        def objective_normal(p_fit_values):
            # Build parameter vector for this model only
            p_full_list = []

            # Build parameter vector maintaining order from self.parameters
            for i, param_name in enumerate(self.parameters):
                if param_name in parameters:
                    # Find which fitted parameter this is
                    fit_idx = param_names.index(param_name)
                    p_full_list.append(p_fit_values[fit_idx])
                else:
                    # Use default value of 1.0 for unfitted parameters
                    print("Parameter in supplied defaults not found: ", param_name)
                    p_full_list.append(1.0)
            
            p_full = np.array(p_full_list, dtype=np.float64)
            
            total_loss = 0.0

            # Process each unique group/experiment
            unique_groups = dataframe['group'].unique()
            for group_id in unique_groups:
                t_func = self.t_func if self.group_to_light == None else self.group_to_light[group_id]
                group_data = dataframe[dataframe['group'] == group_id].sort_values('time')
                
                times = group_data['time'].values.astype(np.float64, copy=False)
                observed_data = group_data['y'].values.astype(np.float64, copy=False)

                meta_cols = [col for col in group_data.columns if col not in {'time', 'y'}]
                group_meta = {col: group_data.iloc[0][col] for col in meta_cols}
                current_t_args = {**base_t_args, **group_meta}

                try:
                    sol = solve_ivp(
                        lambda t, y: system(t, y, p_full, t_func, current_t_args),
                        [times[0], times[-1]], y0, 
                        t_eval=times, method=self.ivp_method, rtol=1e-8
                    )
                    
                    if sol.success:
                        # Use the last state as the observable (in simple model: y, in EGFR: KTR_s) update: ERK now
                        predicted_data = sol.y[-1]  # Last state variable
                        loss = np.sum((predicted_data - observed_data)**2)
                        total_loss += loss
                    else:
                        return 1e10
                
                except Exception as e:
                    print(f"Error solving ODE for group {group_id}: {e}")
                    return 1e10
            
            return total_loss
        
        p_init = [parameters[name] for name in param_names]

        # Choose objective and bounds based on optimizer
        use_log_transform = self.minimizer_method in ['L-BFGS-B', 'trust-constr', 'SLSQP']

        if use_log_transform:
            # Log-space optimization for better numerical stability
            p_init_log = np.log(np.maximum(p_init, 1e-10))  # Avoid log(0)
            objective_func = objective_log
            bounds = [(-10, 5)] * len(p_init)  # exp(-10) to exp(5) ≈ 4.5e-5 to 148
        else:
            objective_func = objective_normal
            bounds = [(1e-6, 100)] * len(p_init)

        # Improved optimizer options
        if self.minimizer_method == 'L-BFGS-B':
            options = {
                'maxiter': 10000,
                'ftol': 1e-9,
                'gtol': 1e-6,
                'maxls': 50
            }
        elif self.minimizer_method == 'trust-constr':
            options = {
                'maxiter': 10000,
                'xtol': 1e-8,
                'gtol': 1e-6
            }
        else:
            options = {'maxiter': 10000}

        initial_params = p_init_log if use_log_transform else p_init

        result = minimize(
            objective_func, initial_params,
            method=self.minimizer_method,
            bounds=bounds,
            options=options
        )
        
        # Package results - transform back from log space if needed
        final_params = np.exp(result.x) if use_log_transform else result.x
        fitted_params = dict(zip(param_names, final_params))

        # Calculate detailed fit statistics
        n_experiments = len(dataframe['group'].unique())
        n_data_points = len(dataframe)
        n_fitted_params = len(param_names)
        degrees_of_freedom = n_data_points - n_fitted_params

        # Per-group statistics
        group_stats = {}
        system = self.make_numerical()

        for group_id in dataframe['group'].unique():
            group_data = dataframe[dataframe['group'] == group_id].sort_values('time')
            t_func = self.t_func if self.group_to_light == None else self.group_to_light[group_id]
            times = group_data['time'].values
            observed_data = group_data['y'].values

            # Simulate with fitted parameters
            p_full_list = []
            for i, param_name in enumerate(self.parameters):
                if param_name in fitted_params:
                    p_full_list.append(fitted_params[param_name])
                else:
                    # Use default value of 1.0 for unfitted parameters
                    print("Parameter in supplied defaults not found: ", param_name)
                    p_full_list.append(1.0)

            p_full = np.array(p_full_list)
            meta_cols = [col for col in group_data.columns if col not in {'time', 'y'}]
            group_meta = {col: group_data.iloc[0][col] for col in meta_cols}
            t_args_copy = {**base_t_args, **group_meta}

            try:
                sol = solve_ivp(
                    lambda t, y: system(t, y, p_full, t_func, t_args_copy),
                    [times[0], times[-1]], y0,
                    t_eval=times, method=self.ivp_method, rtol=1e-8
                )

                if sol.success:
                    predicted_data = sol.y[-1]
                    residuals = predicted_data - observed_data
                    mse = np.mean(residuals**2)
                    rmse = np.sqrt(mse)
                    r_squared = 1 - np.sum(residuals**2) / np.sum((observed_data - np.mean(observed_data))**2)

                    group_stats[str(group_id)] = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r_squared': float(r_squared),
                        'n_points': len(times),
                        'max_observed': float(np.max(observed_data)),
                        'max_predicted': float(np.max(predicted_data))
                    }
            except:
                group_stats[str(group_id)] = {'error': 'simulation_failed'}

        # Overall statistics
        overall_mse = result.fun / n_data_points
        aic = 2 * n_fitted_params + n_data_points * np.log(result.fun / n_data_points) if result.fun > 0 else np.inf
        bic = np.log(n_data_points) * n_fitted_params + n_data_points * np.log(result.fun / n_data_points) if result.fun > 0 else np.inf

        # Build fit result with new structure
        fit_result_data = {
            'fitted_params': fitted_params,
            'loss': result.fun,
            'success': result.success,
            'message': result.message,
            'n_experiments': n_experiments,
            'optimization_info': {
                'method': self.minimizer_method,
                'n_iterations': result.nit if hasattr(result, 'nit') else None,
                'n_function_evaluations': result.nfev if hasattr(result, 'nfev') else None,
                'log_transform_used': use_log_transform,
                'final_gradient_norm': np.linalg.norm(result.jac) if hasattr(result, 'jac') and result.jac is not None else None
            },
            'fit_statistics': {
                'overall_mse': float(overall_mse),
                'rmse': float(np.sqrt(overall_mse)),
                'aic': float(aic),
                'bic': float(bic),
                'degrees_of_freedom': degrees_of_freedom,
                'n_data_points': n_data_points,
                'n_fitted_params': n_fitted_params,
                'group_statistics': group_stats
            },
            'params': {
                "fitted": parameters,
                "all": self.parameters
            }
        }

        # Add config path if parameters were loaded from file
        if config_path is not None:
            fit_result_data['config_path'] = str(config_path)

        self.fit_result = fit_result_data
        return self.fit_result

    def read_config(self, config_path):
        """
        Read parameter configuration from JSON file.

        Handles different JSON structures:
        1. Simple dict: {param_name: value, ...}
        2. Result structure with 'params': {params: {...}}
        3. Result structure with 'fitted_params': {fitted_params: {...}}
        4. Complex nested structures

        Parameters:
        -----------
        config_path : Path or str
            Path to JSON configuration file

        Returns:
        --------
        dict
            Dictionary with {param_name: value} for parameters in self.parameters
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            data = json.load(f)

        # Strategy: try different extraction methods in order of preference
        params_dict = None

        # 1. Try simple dict (most common for pure parameter files)
        if self._is_simple_param_dict(data):
            params_dict = data

        # 2. Try 'fitted_params' key first (most direct from fit results)
        elif 'fitted_params' in data and isinstance(data['fitted_params'], dict):
            params_dict = data['fitted_params']

        # 3. Try 'params' key (common in some result formats)
        elif 'params' in data and isinstance(data['params'], dict):
            # Check if params has nested structure with 'to_fit'
            if 'to_fit' in data['params'] and isinstance(data['params']['to_fit'], dict):
                params_dict = data['params']['to_fit']
            else:
                params_dict = data['params']

        # 4. Try 'meta'.'fitted_params' (nested structure)
        elif 'meta' in data and isinstance(data['meta'], dict) and 'fitted_params' in data['meta']:
            params_dict = data['meta']['fitted_params']

        # 5. Last resort: look for any dict that contains parameter-like keys
        else:
            params_dict = self._extract_params_from_nested(data)

        if params_dict is None:
            raise ValueError(f"Could not extract parameters from {config_path}")

        # Filter to only parameters that exist in self.parameters (maintain order consistency)
        filtered_params = {param: params_dict[param]
                          for param in self.parameters
                          if param in params_dict}

        if not filtered_params:
            available_params = list(params_dict.keys())
            raise ValueError(f"No matching parameters found in config. "
                           f"Config contains: {available_params}, "
                           f"Model expects: {self.parameters}")

        print(f"Loaded {len(filtered_params)} parameters from {config_path}")
        missing_params = set(self.parameters) - set(filtered_params.keys())
        if missing_params:
            print(f"Missing parameters (will use defaults): {missing_params}")

        return filtered_params

    def _is_simple_param_dict(self, data):
        """Check if data is a simple parameter dictionary"""
        if not isinstance(data, dict):
            return False

        # Simple heuristics:
        # - All values are numbers
        # - No nested dicts with metadata-like keys
        metadata_keys = {'fitted_params', 'params', 'meta', 'loss', 'success',
                        'message', 'optimization_info', 'fit_statistics'}

        has_metadata = any(key in data for key in metadata_keys)
        all_numeric = all(isinstance(v, (int, float)) for v in data.values())

        return not has_metadata and all_numeric

    def _extract_params_from_nested(self, data):
        """Extract parameters from nested structure as last resort"""
        if not isinstance(data, dict):
            return None

        # Look for any dict that contains parameter names from self.parameters
        def search_for_params(obj, path=""):
            if isinstance(obj, dict):
                # Check if this dict contains any of our parameter names
                matches = sum(1 for param in self.parameters if param in obj)
                if matches > 0:
                    return obj, matches, path

                # Recursively search nested dicts
                best_match = None
                best_count = 0
                best_path = ""

                for key, value in obj.items():
                    result = search_for_params(value, f"{path}.{key}" if path else key)
                    if result and result[1] > best_count:
                        best_match, best_count, best_path = result

                return (best_match, best_count, best_path) if best_match else None
            return None

        result = search_for_params(data)
        if result:
            params_dict, match_count, path = result
            print(f"Found {match_count} matching parameters at path: {path}")
            return params_dict

        return None

    def save_results(self):
        assert self.fit_result is not None, "To save results you gotta have results."
        time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        path = RESULTS_PATH / f'{self.name}_{time}_{self.minimizer_method}.json'
        from pprint import pprint
        pprint(self.fit_result)
        with open(path, 'w') as f: json.dump(self.fit_result, f, indent=2)