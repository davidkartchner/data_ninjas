import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 4:
    print "usage incsv outarchive test|train"
    exit()

def read_clean_csv(filename):
 data = pd.read_csv(filename)
 pd.set_option('expand_frame_repr', True)
 pd.set_option('max_columns',200)
 other_frames = [data]
 for i in data.columns:
    if data[i].dtype == "object":
     data[i] = data[i].fillna("NA")
     vc = data[i].value_counts()
     class_to_ind = {k: i for i,k in enumerate(vc.keys())}
     if len(vc) < 200:
         dummies = pd.get_dummies(data[i])
         dummies.rename(columns=lambda x: i+"_"+x, inplace=True)
         del data[i]
         other_frames.append(dummies)
#         print pd.concat([data, dummies], axis=1)
     else:
        data[i] = data[i].apply(lambda x: class_to_ind[x]).astype(np.float64)
#         print data.concat(pd.get_dummies(data[i]))
    else:
     data[i] = data[i].fillna(data[i].median())
 res = pd.concat(other_frames, axis=1)
 return res

def train_model(data):
 mat = data.as_matrix()
 nsamples = mat.shape[0]/2
 rf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
 rf.fit(mat[:nsamples,2:], mat[:nsamples,1])
 print "Score on test part of training data",rf.score(mat[nsamples:,2:], mat[nsamples:,1])
 return rf

infile = argv[1]
outfile = argv[2]

d = read_clean_csv(infile).as_matrix()
if argv[3] == "test":
    np.savez(outfile, features = d[:,1:], ids=d[:,0])
else:
    print "assuming train data"
    np.savez(outfile, features=d[:,2:], labels=d[:,1])

#td = d.as_matrix()
#np.savez("train1.npz", features = td[:,2:], labels=td[:,1])
#model = train_model(d)
#test_data = read_clean_csv('test.csv').as_matrix()

#res = model.predict_proba(test_data[:,1:])
#res[:,0] = test_data[:,0] #ids
#np.savetxt("sub2.csv", res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
