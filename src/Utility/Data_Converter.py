# Data_Converter.py
#
# This script converts a Parquet dataset to a CSV file.

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root (im-binclass/)

df = pd.read_parquet(ROOT / 'data' / 'dataset.parquet')
print(df.head(10))
df.to_csv(ROOT / 'data' / 'data_fraunhofer.csv', index=False)