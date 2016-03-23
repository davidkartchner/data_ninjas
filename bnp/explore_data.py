import pandas as pd

data = pd.read_csv("train.csv")
print data['target'][0]

# use data['v22'].value_counts(), data.select_values, data.select_dtypes(include=['object'])
