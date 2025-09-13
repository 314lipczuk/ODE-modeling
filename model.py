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
from jax import grad
import jax.numpy as jnp

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
    def __init__(self, name, states, parameters, model_definition, t_func, t_dep, ivp_method='LSODA', minimizer_method='L-BFGS-B'):
        '''
        We give it a name, we give it all the symbolic equations, and params (defaults?)

        Pipeline:
            data ---------------------------------------.
                                                        |
            eqs + params + light_fn -> numeric system  -L--> make loss function -> feed to minimizer

        What's a good API to insert EQs? Do i need to input symbol for t?           
        '''
        self.name = name
        self.states = states
        self.parameters = parameters
        self.model_definition_f = model_definition
        model = model_definition(parameters, states)
        self.eqs = model['equations']
        self.symbols = model['symbols']
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
            args = [t, *y, *params, res]
            return [f(*args) for f in numerical_funcs]
            
        return system

    def fit(self, dataframe, y0, p0, t_args, params_to_fit, params_to_fix): 
        """
        TODO: simplify, rethink: 
            - do i really need both p0 and params_to_fit?   
            - validity of params_to_fix when i have transforms

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
        p0 : array_like  
            Initial parameter values (fallback for unfitted parameters)
        t_args : list
            Arguments to pass to the time-dependent function (e.g., light intensity)
        params_to_fit : dict
            {param_name: initial_guess} for parameters to optimize
        params_to_fix : dict, optional
            {param_name: fixed_value} for parameters to keep constant
        """
        if params_to_fix is None: 
            params_to_fix = {}
        
        if t_args is None:
            t_args = {}
        else:
            assert type(t_args) == dict, "t_args need to be a dictionary \
                (so i can automatically put generic keys in there, like a group_id)"
        # Validate input dataframe
        required_cols = {'time', 'y', 'group'}
        if not required_cols.issubset(dataframe.columns):
            missing = required_cols - set(dataframe.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.params_to_fit = params_to_fit
        self.params_to_fix = params_to_fix
        param_names = list(params_to_fit.keys())
        
        # Get the numerical system
        system = self.make_numerical()

        def objective_log(p_fit_log_values):
            # Transform from log space back to normal space
            p_fit_values = np.exp(p_fit_log_values)
            return objective_normal(p_fit_values)

        def objective_normal(p_fit_values):
            # Build parameter vector for this model only
            p_full_list = []

            # Start with provided p0 values
            for i, param_name in enumerate(self.parameters):
                if param_name in params_to_fit:
                    # Find which fitted parameter this is
                    fit_idx = param_names.index(param_name)
                    p_full_list.append(p_fit_values[fit_idx])
                elif param_name in params_to_fix:
                    p_full_list.append(params_to_fix[param_name])
                else:
                    # Use default from p0
                    p_full_list.append(p0[i] if i < len(p0) else 1.0)
            
            p_full = np.array(p_full_list)
            
            total_loss = 0.0

            # Process each unique group/experiment
            unique_groups = dataframe['group'].unique()
            for group_id in unique_groups:
                group_data = dataframe[dataframe['group'] == group_id].sort_values('time')
                
                times = group_data['time'].values
                observed_data = group_data['y'].values
                t_args['group'] = group_id
                try:
                    sol = solve_ivp(
                        lambda t, y: system(t, y, p_full, self.t_func, t_args),
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
        
        p_init = [params_to_fit[name] for name in param_names]

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
            times = group_data['time'].values
            observed_data = group_data['y'].values

            # Simulate with fitted parameters
            p_full_list = []
            for i, param_name in enumerate(self.parameters):
                if param_name in fitted_params:
                    p_full_list.append(fitted_params[param_name])
                elif param_name in params_to_fix:
                    p_full_list.append(params_to_fix[param_name])
                else:
                    p_full_list.append(p0[i] if i < len(p0) else 1.0)

            p_full = np.array(p_full_list)
            t_args_copy = t_args.copy() if t_args else {}
            t_args_copy['group'] = group_id

            try:
                sol = solve_ivp(
                    lambda t, y: system(t, y, p_full, self.t_func, t_args_copy),
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

        self.fit_result = {
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
            'params':{
                "to_fix": params_to_fix,
                "to_fit": params_to_fit,
                "all": self.parameters
            }
        }
        return self.fit_result

    def save_results(self):
        assert self.fit_result is not None, "To save results you gotta have results."
        time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        path = RESULTS_PATH / f'{self.name}_{time}_{self.minimizer_method}.json'
        from pprint import pprint
        pprint(self.fit_result)
        with open(path, 'w') as f: json.dump(self.fit_result, f, indent=2)

            