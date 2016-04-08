import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
    exit()

def train_model(features, labels):
 rf = RandomForestClassifier(n_estimators=1000, n_jobs=2)
# rf.fit(mat[:nsamples,2:], mat[:nsamples,1])
 rf.fit(features, labels)
 return rf

outfile = argv[3]
trainfile = argv[1]
testfile = argv[2]

test_arch = np.load(testfile)
train_arch = np.load(trainfile)
train_features = train_arch['features']
train_labels = train_arch['labels']
test_features = test_arch['features']
ids = test_arch['ids']

#assert len(ids) == len(test_features)

model = train_model(train_features, train_labels)
res = np.zeros((len(ids),2))
res[:,1] = model.predict_proba(test_features)
res[:,0] = test_data[:,0] #ids
np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
