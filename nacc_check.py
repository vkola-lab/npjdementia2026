dir = "/projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/datasets/NACC"

import pandas as pd
import os
import json

df = pd.read_csv(os.path.join(dir, "NACC.csv"))
json_file = os.path.join(dir, "NACC.json")

print(df["DRUG2"].unique())