import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

# mmmm spghetti 
df = pd.read_parquet('exp_data.parquet')
df["cnr"] = df["mean_intensity_C1_ring"] / df["mean_intensity_C1_nuc"]
df["uid"] = df["fov"].astype("string") + "_" + df["particle"].astype("string")
df['cell_id'] = df['cell_line'].astype(str) + '_' + df['stim_exposure'].astype(str) + 'ms_' + df['uid'].astype(str)
df["frame"] = df["timestep"]
frame_counts = df["uid"].value_counts()
threshold = 0.9 * frame_counts.max()
valid_uids = frame_counts[frame_counts >= threshold].index
df = df[df["uid"].isin(valid_uids)]
NORM_UNTIL_TIMEPOINT = 10
mean_cnr_first_four_frames = df[df['frame'] < NORM_UNTIL_TIMEPOINT].groupby('uid')['cnr'].mean()
df['cnr_norm'] = df.apply(lambda row: row['cnr'] / mean_cnr_first_four_frames[row['uid']], axis=1)
df["stim_timestep_str"] = df["stim_timestep"].apply(str) 
# Compute frame-to-frame differences
df['diff'] = df.groupby('uid')['cnr'].diff().abs()
# Drop first frame per UID (NaN in diff)
df = df.dropna(subset=['diff'])
# Compute mean absolute difference per UID
df['mean_diff'] = df.groupby('uid')['diff'].transform('mean')
# Define a threshold (e.g., remove top 0.02% fluctuating cells)
threshold = df['mean_diff'].quantile(0.998)
df_filtered = df[df['mean_diff'] < threshold]
# Flag cells that will be deleted
df['is_deleted'] = df['mean_diff'] >= threshold
# Display the rows that are deleted
deleted_rows = df[df['is_deleted']]



STIM_TIMESTEP_TO_PLOT = "[10]"
df_plot = df.query("stim_timestep_str == @STIM_TIMESTEP_TO_PLOT")
cell_lines = df_plot["cell_line"].unique()
df_plot.loc[:, "stim_exposure"] = df_plot.loc[:, "stim_exposure"].astype("int")

cell_line = 'EGFR'
plt.figure(figsize=(6, 3),dpi = 250)
num_cells = df_plot.query("cell_line == @cell_line")["uid"].nunique()

sns.lineplot(data=df_plot.query("cell_line == @cell_line"), x='frame', y='cnr_norm', hue="stim_exposure", palette="tab10", estimator="median", errorbar=('ci',90))
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = []
for label in labels:
    exposure_time = int(float(label))
    num_cells_exposure = df_plot.query("cell_line == @cell_line and stim_exposure == @exposure_time and stim_timestep_str == @STIM_TIMESTEP_TO_PLOT")["uid"].nunique()
    new_labels.append(f'{int(float(label))} ms (n={num_cells_exposure})')
plt.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
plt.ylabel('norm. ERK-KTR c/n ratio')
plt.xlabel('time [min]')
plt.axvline(x=10, color='black', linestyle='--')
plt.title(f"{cell_line} (n={num_cells})")
plt.savefig(f"{cell_line}_single_erkktr_norm.svg", bbox_inches='tight')
plt.show()


#df_plot[df_plot['cell_line'] == 'EGFR'] # cnr norm, by frame (each frame is 1 minute);

# To model, at the beginning
# for EGFR cell line,
# per each frame and light exposure time
# get a median of the value of normalized cnr

to_model = df_plot[df_plot['cell_line'] == 'EGFR'].groupby(['frame','stim_exposure' ])['cnr_norm'].median().reset_index()
exposure_types = df_plot[df_plot['cell_line'] == 'EGFR']['stim_exposure'].unique()

# now i can just access each exp type by 
to_model[to_model['stim_exposure']==200]

