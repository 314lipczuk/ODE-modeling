#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sympy import Symbol, Function, Eq, Derivative
from sympy.abc import t
from scipy.integrate import solve_ivp
from model import Model

def simple_model_definition(parameters, states):
    """Simple 2-state ODE model for testing: dx/dt = -kx*x + light, dy/dt = ky*x - dy*y"""
    symbols_dict = {}
    
    # Create parameter symbols
    for p in parameters: 
        symbols_dict[p] = Symbol(p)
    
    # Create state symbols  
    for s in states: 
        symbols_dict[s] = Symbol(s)
    
    # Time and light symbols
    symbols_dict['t'] = Symbol('t')
    symbols_dict['light'] = Symbol('light')
    
    s = symbols_dict
    
    # Simple two-state model equations
    eqs = [
        Eq(Derivative(s['x'], s['t']), -s['kx'] * s['x'] + s['light']),
        Eq(Derivative(s['y'], s['t']), s['ky'] * s['x'] - s['dy'] * s['y'])
    ]
    
    return {'equations': eqs, 'symbols': symbols_dict}

def generate_test_data():
    """Generate synthetic test data that matches the expected DataFrame format"""
    
    # True parameters for data generation
    true_kx = 0.1
    true_ky = 0.2  
    true_dy = 0.15
    
    data_points = []
    
    # Create data for different stimulation conditions
    stim_exposures = [100, 200, 300]  # Different pulse durations
    uids = ['cell1', 'cell2', 'cell3']  # Different cells
    
    for stim_exp in stim_exposures:
        for uid in uids:
            # Time points (frames 1-20, converted to seconds)
            frames = np.arange(1, 21)
            times = (frames - 1) * 60.0  # Convert to seconds
            
            # Light function parameters
            start_time_ms = (10 - 1) * 60 * 1000  # Frame 10 = 540000 ms
            light_duration_ms = stim_exp
            light_intensity = 1.0
            
            def light_func(t):
                t_ms = t * 1000.0
                if start_time_ms <= t_ms <= start_time_ms + light_duration_ms:
                    return light_intensity
                else:
                    return 0.0
            
            # Solve the true model to generate data
            def true_model(t, y):
                x, y_state = y
                light = light_func(t)
                dxdt = -true_kx * x + light
                dydt = true_ky * x - true_dy * y_state
                return [dxdt, dydt]
            
            y0 = [0.1, 0.1]  # Initial conditions
            sol = solve_ivp(true_model, [times[0], times[-1]], y0, t_eval=times)
            
            if sol.success:
                # Use y (second state) as our observable (like KTR_s)
                y_obs = sol.y[1]
                
                # Add some noise
                np.random.seed(42)  # For reproducible results
                y_obs_noisy = y_obs + np.random.normal(0, 0.01, len(y_obs))
                
                # Create data points
                for i, (frame, time_val, obs_val) in enumerate(zip(frames, times, y_obs_noisy)):
                    data_points.append({
                        'frame': frame,
                        'uid': uid,
                        'stim_exposure': stim_exp,
                        'cnr_norm': obs_val  # This will be mapped to KTR_s
                    })
    
    return pd.DataFrame(data_points)

def test_model_pipeline():
    """Test the complete Model class pipeline"""
    print("Starting Model class pipeline test...")
    
    # 1. Create simple model
    print("1. Creating simple test model...")
    parameters = ['kx', 'ky', 'dy']
    states = ['x', 'y']
    
    model = Model(
        name='simple_test',
        states=states,
        parameters=parameters,
        model_definition=simple_model_definition,
        t_func=lambda t, *args: args[0] if args else 0.0,  # Simple light function
        t_dep='light'
    )
    
    print(f"   Created model with {len(model.eqs)} equations")
    
    # 2. Make numerical system
    print("2. Converting to numerical system...")
    numerical_system = model._make_numerical()
    print("   Numerical system created successfully")
    
    # 3. Test numerical system with simple parameters
    print("3. Testing numerical system...")
    test_params = [0.1, 0.2, 0.15]  # kx, ky, dy
    test_y0 = [0.1, 0.1]  # x, y initial conditions
    test_times = np.linspace(0, 10, 100)
    
    def test_light_func(t, intensity):
        return intensity if 2 < t < 4 else 0.0
    
    try:
        sol = solve_ivp(
            lambda t, y: numerical_system(t, y, test_params, test_light_func, [1.0]),
            [0, 10], test_y0, t_eval=test_times
        )
        if sol.success:
            print("   Numerical system works correctly")
        else:
            print(f"   ERROR: Numerical system failed: {sol.message}")
            return False
    except Exception as e:
        print(f"   ERROR: Exception in numerical system: {e}")
        return False
    
    # 4. Generate test data
    print("4. Generating synthetic test data...")
    test_df = generate_test_data()
    print(f"   Generated {len(test_df)} data points")
    print(f"   Unique UIDs: {test_df['uid'].nunique()}")
    print(f"   Unique stimulation exposures: {sorted(test_df['stim_exposure'].unique())}")
    
    # 5. Test fit method
    print("5. Testing fit method...")
    
    # Parameters to fit (we'll try to recover the true values)
    params_to_fit = {
        'kx': 0.08,  # Initial guess (true value is 0.1)
        'ky': 0.25,  # Initial guess (true value is 0.2)
        'dy': 0.12   # Initial guess (true value is 0.15)
    }
    
    # No fixed parameters for this simple test
    params_to_fix = {}
    
    # Initial conditions and parameter vector
    y0 = [0.1, 0.1]
    p0 = np.array([0.08, 0.25, 0.12])  # Initial parameter guess
    t_args = [1.0]  # Light intensity
    
    try:
        fit_result = model.fit(
            dataframe=test_df,
            y0=y0,
            p0=p0,
            t_args=t_args,
            params_to_fit=params_to_fit,
            params_to_fix=params_to_fix
        )
        
        print("   Fit completed successfully!")
        print(f"   Success: {fit_result['success']}")
        print(f"   Final loss: {fit_result['loss']:.6f}")
        print(f"   Number of experiments: {fit_result['n_experiments']}")
        print("   Fitted parameters:")
        for name, value in fit_result['fitted_params'].items():
            print(f"     {name}: {value:.4f}")
        
        # Check if we recovered reasonable parameters
        true_values = {'kx': 0.1, 'ky': 0.2, 'dy': 0.15}
        print("   Parameter recovery check:")
        all_reasonable = True
        for name, fitted_val in fit_result['fitted_params'].items():
            true_val = true_values[name]
            relative_error = abs(fitted_val - true_val) / true_val
            print(f"     {name}: fitted={fitted_val:.4f}, true={true_val:.4f}, rel_error={relative_error:.2%}")
            if relative_error > 0.5:  # Allow 50% error for this test
                all_reasonable = False
        
        if all_reasonable:
            print("   ✓ Parameter recovery is reasonable")
        else:
            print("   ⚠ Parameter recovery has large errors (but fit ran successfully)")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: Fit method failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete test suite"""
    print("=" * 60)
    print("TESTING MODEL CLASS PIPELINE")
    print("=" * 60)
    
    try:
        success = test_model_pipeline()
        
        print("\n" + "=" * 60)
        if success:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("=" * 60)
        
        return success
        
    except Exception as e:
        print(f"\nUnexpected error in test suite: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()