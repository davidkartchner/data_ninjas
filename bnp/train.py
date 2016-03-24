from sklearn.Ensemble import RandomForestClassifier
import pandas as pd

def remove_nans(arr):
 nan_mask = np.isnan(arr)
 med = np.median(arr[np.logical_not(nan_mask)])
 arr[nan_mask] = med

def cleanup_data():
 
#rf = RandomForestClassifier(n_estimators = 100)
#rf.fit(train, target)

