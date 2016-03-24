import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt


def read_clean_csv(filename):
 data = pd.read_csv(filename)
 pd.set_option('expand_frame_repr', True)
 pd.set_option('max_columns',200)
 for i in data.select_dtypes(include=['object'])[:0]:
    data[i] = data[i].fillna("NA")   
    vc = data[i].value_counts()
    class_to_ind = {k: i for i,k in enumerate(vc.keys())}
    data[i] = data[i].apply(lambda x: class_to_ind[x]).astype(np.float64)

 for i in data.select_dtypes(exclude=['category'])[:0]:
    data[i] = data[i].fillna(data[i].median())
 return data

def train_model(data):
 mat = data.as_matrix()
 nsamples = mat.shape[0]/2
 rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
 rf.fit(mat[:nsamples,2:], mat[:nsamples,1])
 print "Score on test part of training data",rf.score(mat[nsamples:,2:], mat[nsamples:,1])
 return rf

d = read_clean_csv('train.csv')
model = train_model(d)
test_data = read_clean_csv('test.csv').as_matrix()
res = model.predict_proba(test_data[:,1:])
res[:,0] = test_data[:,0] #ids
np.savetxt("sub2.csv", res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
