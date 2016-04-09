import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
    exit()

def train_model(features, labels):
 rf = RandomForestClassifier(n_estimators=500, criterion= "entropy",n_jobs=1, max_depth = 35, min_samples_split=4, min_samples_leaf = 2)
# rf.fit(mat[:nsamples,2:], mat[:nsamples,1])
 rf.fit(features, labels)
 return rf

def  validate_model(model, features, labels):
 print cross_val_score(model, features, labels, scoring="log_loss")


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
print "loaded data"
model = train_model(train_features, train_labels)
print "Trained"
res = np.zeros((len(ids),2))
#print model.predict_proba(test_features)
res[:,1] = model.predict_proba(test_features)[:,1]
res[:,0] = ids #ids
np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
print "validating"
validate_model(model, train_features, train_labels)
