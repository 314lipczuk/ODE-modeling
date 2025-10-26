from pathlib import Path, PosixPath
import pandas as pd
import json
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT / 'data'
UTILS_PATH = PROJECT_ROOT / 'utils'
MODELS_PATH = PROJECT_ROOT / 'models'  
RESULTS_PATH = PROJECT_ROOT / 'results'

# Additional paths that might be useful
VENV_PATH = PROJECT_ROOT / 'venv'

def ensure_dirs_exist():
    """Create directories if they don't exist"""
    for path in [DATA_PATH, MODELS_PATH, RESULTS_PATH ]:
        path.mkdir(exist_ok=True, parents=True)

def get_data_file(filename):
    """Get path to a file in the data directory"""
    return DATA_PATH / filename

def get_results_file(filename):
    """Get path to a file in the results directory"""
    return RESULTS_PATH / filename

def get_model_file(filename):
    """Get path to a file in the models directory"""
    return MODELS_PATH / filename

# Print paths for debugging (can be removed later)
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"UTILS_PATH: {UTILS_PATH}")
    print(f"MODELS_PATH: {MODELS_PATH}")
    print(f"RESULTS_PATH: {RESULTS_PATH}")
    print()
    print("Path existence check:")
    for name, path in [("DATA", DATA_PATH), ("UTILS", UTILS_PATH), 
                       ("MODELS", MODELS_PATH), ("RESULTS", RESULTS_PATH),
                       ]:
        print(f"  {name}: {path.exists()}")


def read_config(config_path, parameters):
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

    assert parameters is not None
    assert Path(config_path).is_file()

    def _is_simple_param_dict(data):
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

    def _extract_params_from_nested(data, parameters=parameters):
        """Extract parameters from nested structure as last resort"""
        if not isinstance(data, dict):
            return None

        # Look for any dict that contains parameter names from self.parameters
        def search_for_params(obj, parameters=parameters, path=""):
            if isinstance(obj, dict):
                # Check if this dict contains any of our parameter names
                matches = sum(1 for param in parameters if param in obj)
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


    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        data = json.load(f)

    # Strategy: try different extraction methods in order of preference
    params_dict = None

    # 1. Try simple dict (most common for pure parameter files)
    if _is_simple_param_dict(data):
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
        params_dict = _extract_params_from_nested(data)

    if params_dict is None:
        raise ValueError(f"Could not extract parameters from {config_path}")

    # Filter to only parameters that exist in self.parameters (maintain order consistency)
    filtered_params = {param: params_dict[param]
                      for param in parameters
                      if param in params_dict}

    if not filtered_params:
        available_params = list(params_dict.keys())
        raise ValueError(f"No matching parameters found in config. "
                       f"Config contains: {available_params}, "
                       f"Model expects: {parameters}")

    print(f"Loaded {len(filtered_params)} parameters from {config_path}")
    missing_params = set(parameters) - set(filtered_params.keys())
    if missing_params:
        print(f"Missing parameters (will use defaults): {missing_params}")

    return filtered_params

def generate_uniform_dist_param(params, name):
  param_len = len(params)
  print('Generating uniformly distributed parameter set of length', param_len)
  n = np.random.uniform(0.01, 2, (param_len,))
  new_params = {k:n[i] for (i,k) in enumerate(params)}
  with open(RESULTS_PATH / name, 'w') as f : json.dump(new_params, f)
  return new_params