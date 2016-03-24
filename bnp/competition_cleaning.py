#David Kartchner
#March 24, 2016

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

pd.set_option('expand_frame_repr', True)
pd.set_option('max_columns',200)
data[:10]
for i in data.select_dtypes(include=['object'])[:0]:
    data[i] = data[i].astype('category')

for i in data.select_dtypes(exclude=['category'])[:0]:
    data[i] = data[i].fillna(data[i].median())

