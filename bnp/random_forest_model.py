import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
    exit()

def build_model():
 return RandomForestClassifier(n_estimators=1000, criterion= "entropy",n_jobs=-1, max_depth = 35, min_samples_split=4, min_samples_leaf = 2)

def train_model(features, labels):
 rf = build_model()
# rf = RandomForestClassifier(n_estimators=1000, criterion= "entropy",n_jobs=1, max_depth = 50, min_samples_split=4, min_samples_leaf = 2)
# rf.fit(mat[:nsamples,2:], mat[:nsamples,1])
 rf.fit(features, labels)
 return rf

def  validate_model(features, labels):
 model = build_model()
 print cross_val_score(model, features, labels, scoring="log_loss")

def cross_val_model(features, labels, test_features, n_fold=5):
    skf = StratifiedKFold(labels, n_folds=n_fold)
    probs = np.zeros(len(test_features))
    i=0
    for train_mask, test_mask in skf:
        model = train_model(features[train_mask], labels[train_mask])
        probs += model.predict_proba(test_features)[:,1]
        i+=1
        print i
    probs /= n_fold
    return probs

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
#print "loaded data"
#model = train_model(train_features, train_labels)
#print "Trained"
#validate_model(train_features, train_labels)
res = np.zeros((len(ids),2))
#print model.predict_proba(test_features)
res[:,1] = cross_val_model(train_features, train_labels, test_features)
res[:,0] = ids #ids
np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
#print "validating"
#validate_model(model, train_features, train_labels)
