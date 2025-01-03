from ucimlrepo import fetch_ucirepo
import pandas as pd
import json
with open("config.json", 'r') as file:
    config = json.load(file)

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
Y = wine_quality.data.targets

# Save to CSV file
df = pd.concat([Y, X], axis=1)
df.to_csv(config["data"]["file_name"], index=False)
