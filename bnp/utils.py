from sys import argv, exit
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold

def read_data():
    if len(argv) < 4:
        print "Usage trainnpz testnpz outfile"
        exit()

    outfile = argv[-1]
    trainfile = argv[1]
    testfile = argv[2]

    test_arch = np.load(testfile)
    train_arch = np.load(trainfile)
    train_features = train_arch['features']
    train_labels = train_arch['labels']
    test_features = test_arch['features']
    ids = test_arch['ids']
    return train_features, train_labels, test_features, ids, outfile

def save_submission(filename, ids=None, probs=None):
    pd.DataFrame({"ID":ids, "PredictedProb":probs}).to_csv(filename, index=False)
    
    
def cross_val_model(model, train_features, labels, test_features, nfolds = 5, return_trainprobs = False):
    skf = StratifiedKFold(labels, n_folds=nfolds, random_state=int(time()))
    probs = np.zeros(len(test_features))
    trainprobs = np.zeros(len(train_features))
    i=0
    for train_mask, test_mask in skf:
        i+=1
        model.fit(train_features[train_mask], labels[train_mask])
        probs += model.predict_proba(test_features)[:,1]
        trainprobs[test_mask] = model.predict_proba(train_features[test_mask])[:,1]
        print "Finished cross val fold %d with val error %f" % (i, log_loss(labels[test_mask], trainprobs[test_mask] ))
    probs /= nfolds
    if return_trainprobs:
      return probs, trainprobs
    return probs


