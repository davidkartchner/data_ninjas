import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
    exit()

def xgb_params():
    params = {}
    params['eta'] = .2
    params['objective'] = 'binary:logistic'
    params['silent'] = 1
    params["eval_metric"] = 'logloss'
#    params['lambda'] = 10
#    params['alpha'] = 10
#    params['max_depth'] = 10
#    params['colsample_bytree'] = .5
    return params
#    params[]
num_rounds = 50
def train_model(features, labels):
 xgmat = xgb.DMatrix(features, label=labels)
 #create params
 params = xgb_params()

 bst = xgb.train(params, xgmat, num_rounds, [(xgmat, 'train')])
 return bst

def  validate_model(features, labels):
 xgmat = xgb.DMatrix(features, label=labels)
 #create params
 params = xgb_params()

 print xgb.cv(params, xgmat, num_rounds, metrics=["logloss"])


outfile = argv[3]
trainfile = argv[1]
testfile = argv[2]

test_arch = np.load(testfile)
train_arch = np.load(trainfile)
train_features = train_arch['features']
train_labels = train_arch['labels']
test_features = test_arch['features']
ids = test_arch['ids']
print "loaded data"

def train_and_save():
 model = train_model(train_features, train_labels)
 #assert len(ids) == len(test_features)
 model.save_model("xgboost_bnp4.model")
 print "Trained"
 res = np.zeros((len(ids),2))
 res[:,1] = model.predict(xgb.DMatrix(test_features))
 #print model.predict_proba(test_features)
 #res[:,1] = model.predict_proba(test_features)[:,1]
 res[:,0] = ids #ids
 np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")

print "validating"
validate_model(train_features, train_labels)
