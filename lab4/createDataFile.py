from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
Y = breast_cancer_wisconsin_diagnostic.data.targets
Y = Y.replace({'M': 1, 'B': 0})

# Save to CSV file
df = pd.concat([Y, X], axis=1)
df.to_csv("data.csv", index=False)
