from sys import argv, exit
import numpy as np
import pandas as pd

def read_data():
    if len(argv) < 4:
        print "Usage trainnpz testnpz outfile"
        exit()

    outfile = argv[3]
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
