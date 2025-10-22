dir = "/projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/datasets/NACC"

import pandas as pd
import os
import json

df = pd.read_csv(os.path.join(dir, "NACC.csv"))
json_file = os.path.join(dir, "NACC.json")

# Load the list of keys from the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)
    if isinstance(data, dict):
        json_keys = set(data.keys())
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        # If the JSON is a list of dicts, take the union of all keys in all dicts
        json_keys = set()
        for d in data:
            json_keys.update(d.keys())
    else:
        raise ValueError("Unexpected JSON structure in {}".format(json_file))

# Keep only columns that are present in the JSON keys
df = df[[col for col in df.columns if col in json_keys]]
print(len(df.columns))
df.to_csv(os.path.join(dir, "NACC_filtered.csv"), index=False)