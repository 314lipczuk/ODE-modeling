import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import seaborn as sns
    import os
    from utils.utils import RESULTS_PATH
    from models.simple_EGFR_transient import m
    return Path, RESULTS_PATH, json, m, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Parameter Comparison Tool

    Compare parameters from fit results in two different modes:
    1. Compare a config file against its default parameter starting point
    2. Compare two different config files from the results directory
    """
    )
    return


@app.cell
def _(RESULTS_PATH, mo, os):
    # Comparison mode selection
    comparison_mode = mo.ui.radio(
        options={
            "Config vs Default Parameters": "config_vs_default",
            "Compare Two Config Files": "config_vs_config"
        },
        label="Select comparison mode"
    )

    # Get available JSON files from results directory
    json_files = sorted([f for f in os.listdir(RESULTS_PATH) if f.endswith('.json')],
                       key=lambda f: os.path.getctime(RESULTS_PATH / f), reverse=True)

    comparison_mode
    return comparison_mode, json_files


@app.cell
def _(comparison_mode, json_files, mo):
    # Always create both sets of UI elements but only display the relevant ones
    config_selector = mo.ui.dropdown(
        options=json_files,
        value=json_files[0] if json_files else None,
        label="Select config file to compare with defaults"
    )

    config1_selector = mo.ui.dropdown(
        options=json_files,
        value=json_files[0] if json_files else None,
        label="Select first config file"
    )
    config2_selector = mo.ui.dropdown(
        options=json_files,
        value=json_files[1] if len(json_files) > 1 else (json_files[0] if json_files else None),
        label="Select second config file"
    )

    if comparison_mode.value == "config_vs_default":
        selectors_display = mo.vstack([
            mo.md("**Selected mode:** Config vs Default Parameters"),
            config_selector
        ])
    else:  # config_vs_config
        selectors_display = mo.vstack([
            mo.md("**Selected mode:** Compare Two Config Files"),
            mo.hstack([config1_selector, config2_selector], justify="start")
        ])

    selectors_display
    return config1_selector, config2_selector, config_selector


@app.cell
def _(
    Path,
    RESULTS_PATH,
    comparison_mode,
    config1_selector,
    config2_selector,
    config_selector,
    json,
    json_files,
    m,
    mo,
    np,
    pd,
):
    if not json_files:
        result = mo.md("No JSON files found in results directory.")
    else:
        try:
            if comparison_mode.value == "config_vs_default":
                # Get the selected file from UI element
                selected_file = config_selector.value if config_selector.value else json_files[0]

                # Load config using read_config to get fitted parameters
                params1 = m.read_config(RESULTS_PATH / selected_file)

                # Try to load default/starting parameters from the same config file
                try:
                    with open(RESULTS_PATH / selected_file, 'r') as f:
                        config_data = json.load(f)

                    # Try to extract starting parameters from various possible locations
                    starting_params = None
                    default_source = None

                    # First check if there's a config_path pointing to original starting parameters
                    if 'config_path' in config_data:
                        try:
                            with open(config_data['config_path'], 'r') as config_f:
                                original_config = json.load(config_f)
                            starting_params = original_config  # Assume it's a simple parameter dict
                            default_source = f"Starting Parameters (from config_path: {Path(config_data['config_path']).name})"
                        except Exception:
                            pass  # Fall through to other methods

                    # Check for params.to_fit (initial values before optimization)
                    if starting_params is None and ('params' in config_data and
                        isinstance(config_data['params'], dict) and
                        'to_fit' in config_data['params']):
                        starting_params = config_data['params']['to_fit']
                        default_source = "Starting Parameters (from params.to_fit)"

                    # Check for other possible starting parameter locations
                    elif starting_params is None and 'initial_params' in config_data:
                        starting_params = config_data['initial_params']
                        default_source = "Starting Parameters (from initial_params)"

                    elif starting_params is None and 'p0' in config_data:
                        starting_params = config_data['p0']
                        default_source = "Starting Parameters (from p0)"

                    # Filter to only include parameters that exist in our model
                    if starting_params:
                        params2 = {param: starting_params[param]
                                  for param in m.parameters
                                  if param in starting_params}
                        file2_name = default_source

                        if len(params2) < len(m.parameters):
                            # Some parameters missing from config, fill with 1.0
                            missing_count = len(m.parameters) - len(params2)
                            for param in m.parameters:
                                if param not in params2:
                                    params2[param] = 1.0
                            file2_name += f" + {missing_count} defaults (1.0)"
                    else:
                        # No starting parameters found, use defaults
                        params2 = {param: 1.0 for param in m.parameters}
                        file2_name = "Default Parameters (1.0) - no starting params found in config"

                except Exception:
                    # If we can't read the file or parse it, fall back to defaults
                    params2 = {param: 1.0 for param in m.parameters}
                    file2_name = "Default Parameters (1.0) - config read error"

                file1_name = f"{selected_file} (fitted)"

            else:  # config_vs_config
                # Get selected files from UI elements
                file1 = config1_selector.value if config1_selector.value else json_files[0]
                file2 = config2_selector.value if config2_selector.value else (json_files[1] if len(json_files) > 1 else json_files[0])

                # Load both configs using read_config
                params1 = m.read_config(RESULTS_PATH / file1)
                params2 = m.read_config(RESULTS_PATH / file2)

                file1_name = file1
                file2_name = file2

            # Find common parameters and differences
            common_params = set(params1.keys()) & set(params2.keys())
            only_in_1 = set(params1.keys()) - set(params2.keys())
            only_in_2 = set(params2.keys()) - set(params1.keys())

            comparison_data = []
            for param in sorted(common_params):
                val1 = params1[param]
                val2 = params2[param]
                diff = val2 - val1
                rel_diff = (diff / val1) * 100 if val1 != 0 else np.inf

                comparison_data.append({
                    'Parameter': param,
                    file1_name: val1,
                    file2_name: val2,
                    'Absolute Difference': diff,
                    'Relative Difference (%)': rel_diff
                })

            df = pd.DataFrame(comparison_data)

            result = {
                'df': df,
                'common_params': common_params,
                'only_in_1': only_in_1,
                'only_in_2': only_in_2,
                'params1': params1,
                'params2': params2,
                'file1_name': file1_name,
                'file2_name': file2_name,
                'comparison_mode': comparison_mode.value
            }

        except Exception as e:
            result = mo.md(f"Error processing files: {str(e)}")

    result
    return (result,)


@app.cell
def _(mo, result):
    if not isinstance(result, dict):
        info_display = result
    else:
        info_df = result['df']
        info_common_params = result['common_params']
        info_only_in_1 = result['only_in_1']
        info_only_in_2 = result['only_in_2']
        info_file1_name = result['file1_name']
        info_file2_name = result['file2_name']
        info_mode = result['comparison_mode']

        mode_desc = "Config vs Default" if info_mode == "config_vs_default" else "Config vs Config"

        # Add special note for config vs default mode
        if info_mode == "config_vs_default":
            if "Starting Parameters (from params.to_fit)" in info_file2_name:
                mode_note = "📊 **Comparing fitted vs starting parameters** - showing optimization progress!"
            elif "Default Parameters (1.0)" in info_file2_name:
                mode_note = "⚠️ **Using default values (1.0)** - no starting parameters found in config."
            else:
                mode_note = ""
        else:
            mode_note = ""

        info_text = f"""
        **Comparison Summary ({mode_desc}):**
        {mode_note}

        - Common parameters: {len(info_common_params)}
        - Only in {info_file1_name}: {len(info_only_in_1)} ({', '.join(sorted(info_only_in_1)) if info_only_in_1 else 'none'})
        - Only in {info_file2_name}: {len(info_only_in_2)} ({', '.join(sorted(info_only_in_2)) if info_only_in_2 else 'none'})

        **Debug Info:**
        - Total parameters in {info_file1_name}: {len(result['params1'])}
        - Total parameters in {info_file2_name}: {len(result['params2'])}
        """

        info_display = mo.md(info_text)

    info_display
    return


@app.cell
def _(mo, result):
    if not isinstance(result, dict):
        table_display = mo.md("")
    else:
        table_df = result['df']
        table_display = mo.ui.table(table_df, selection=None)

    table_display
    return


@app.cell
def _(mo, np, plt, result):
    if not isinstance(result, dict):
        viz_display = mo.md("")
    else:
        viz_df = result['df']
        viz_file1_name = result['file1_name']
        viz_file2_name = result['file2_name']

        if len(viz_df) == 0:
            viz_display = mo.md("No common parameters to visualize.")
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            viz_params = viz_df['Parameter'].values
            viz_vals1 = viz_df[viz_file1_name].values
            viz_vals2 = viz_df[viz_file2_name].values
            viz_abs_diff = viz_df['Absolute Difference'].values
            viz_rel_diff = viz_df['Relative Difference (%)'].values

            x_pos = np.arange(len(viz_params))
            width = 0.35

            # Bar chart comparison
            ax1.bar(x_pos - width/2, viz_vals1, width, label=viz_file1_name, alpha=0.8)
            ax1.bar(x_pos + width/2, viz_vals2, width, label=viz_file2_name, alpha=0.8)
            ax1.set_xlabel('Parameters')
            ax1.set_ylabel('Values')
            ax1.set_title('Parameter Values Comparison')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(viz_params, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Scatter plot
            ax2.scatter(viz_vals1, viz_vals2, alpha=0.7, s=50)
            viz_lims = [
                np.min([ax2.get_xlim(), ax2.get_ylim()]),
                np.max([ax2.get_xlim(), ax2.get_ylim()]),
            ]
            ax2.plot(viz_lims, viz_lims, 'k--', alpha=0.5, zorder=0)
            ax2.set_xlabel(f'{viz_file1_name} values')
            ax2.set_ylabel(f'{viz_file2_name} values')
            ax2.set_title('Parameter Values Correlation')
            ax2.grid(True, alpha=0.3)

            # Absolute differences
            viz_colors = ['red' if x > 0 else 'blue' for x in viz_abs_diff]
            ax3.bar(x_pos, viz_abs_diff, color=viz_colors, alpha=0.7)
            ax3.set_xlabel('Parameters')
            ax3.set_ylabel('Absolute Difference')
            ax3.set_title(f'Absolute Differences ({viz_file2_name} - {viz_file1_name})')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(viz_params, rotation=45, ha='right')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3)

            # Relative differences
            viz_rel_diff_finite = np.where(np.isfinite(viz_rel_diff), viz_rel_diff, 0)
            viz_colors_rel = ['red' if x > 0 else 'blue' for x in viz_rel_diff_finite]
            ax4.bar(x_pos, viz_rel_diff_finite, color=viz_colors_rel, alpha=0.7)
            ax4.set_xlabel('Parameters')
            ax4.set_ylabel('Relative Difference (%)')
            ax4.set_title(f'Relative Differences ({viz_file2_name} - {viz_file1_name})')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(viz_params, rotation=45, ha='right')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            viz_display = plt.gca()

    viz_display
    return


@app.cell
def _(mo, result):
    if not isinstance(result, dict):
        summary_display = mo.md("")
    else:
        summary_df = result['df']
        summary_file1_name = result['file1_name']
        summary_file2_name = result['file2_name']

        if len(summary_df) == 0:
            summary_display = mo.md("No common parameters found between the two files.")
        else:
            summary_largest_abs_diff = summary_df.loc[summary_df['Absolute Difference'].abs().idxmax()]
            summary_largest_rel_diff = summary_df.loc[summary_df['Relative Difference (%)'].abs().idxmax()]

            summary_text = f"""
            **Key Differences:**

            **Largest Absolute Difference:**
            - Parameter: `{summary_largest_abs_diff['Parameter']}`
            - Difference: {summary_largest_abs_diff['Absolute Difference']:.4f}
            - Values: {summary_largest_abs_diff[summary_file1_name]:.4f} → {summary_largest_abs_diff[summary_file2_name]:.4f}

            **Largest Relative Difference:**
            - Parameter: `{summary_largest_rel_diff['Parameter']}`
            - Relative Change: {summary_largest_rel_diff['Relative Difference (%)']:.2f}%
            - Values: {summary_largest_rel_diff[summary_file1_name]:.4f} → {summary_largest_rel_diff[summary_file2_name]:.4f}
            """

            summary_display = mo.md(summary_text)

    summary_display
    return


if __name__ == "__main__":
    app.run()
