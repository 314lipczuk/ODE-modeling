import pandas as pd
from utils.utils import *

parquet_file = DATA_PATH / 'exp_data.parquet'
normalized_df = prep_from_parquet(parquet_file, 'data_transient_v3.csv')
