from pathlib import Path

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

