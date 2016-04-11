import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
from sys import argv, exit

if len(argv) < 5:
    print "usage traincsv testcsv trainarchive testarchive"
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
     else:
        data[i] = data[i].apply(lambda x: class_to_ind[x]).astype(np.float64)
    else:
     data[i] = data[i].fillna(-999)
 res = pd.concat(other_frames, axis=1)
 return res

# def analyze_dummies(df):
#     dcols, ddict = [], {}
#     for col in df.columns:
#         if df[col].dtype == "object":
#             df[col] = df[col].fillna("NA")
#             vc = df[col].value_counts()
#             if len(vc) < 200:
#                 dcols.append(col)
#                 ddict[col] = vc.keys()
#                 cols = pd.get_dummies(df[col]).columns
#     return dcols, ddict

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 


def parse_train(filename):
 data = pd.read_csv(filename)
 pd.set_option('expand_frame_repr', True)
 other_frames = [data]
 dummy_dict = {}
 for i in data.columns:
    if data[i].dtype == "object":
         data[i] = data[i].fillna("NA")
         vc = data[i].value_counts()
         class_to_ind = {k: i for i,k in enumerate(vc.keys())}
         if len(vc) < 200:
             dummies = pd.get_dummies(data[i])
             dummy_dict[i] = dummies.columns
             dummies.rename(columns=lambda x: i+"_"+x, inplace=True)
             other_frames.append(dummies)

             del data[i]
         data[i] = pd.factor(data[i])
#        data[i] = data[i].apply(lambda x: class_to_ind[x]).astype(np.float64)
    else:
    
     data[i] = data[i].fillna(-999)
#     data[i+"_inv"] = invert(data[i])
 res = pd.concat(other_frames, axis=1)
 return res, dummy_dict

def parse_test(filename, dummy_dict):
    data = pd.read_csv(filename)
    pd.set_option('expand_frame_repr', True)
    pd.set_option('max_columns',200)
    other_frames = [data]
    for i in data.columns:
        if data[i].dtype == "object":
            dummies = pd.DataFrame()
            if i in dummy_dict:
                data[i] = data[i].fillna("NA")
                dummies = pd.DataFrame()
                cols = dummy_dict[i]
                for col in cols:
                    dummies[col] = (data[i] == col)
                    dummies.rename(columns=lambda x: i+"_"+x, inplace=True)
                other_frames.append(dummies)

                del data[i]
            data[i] = pd.factor(data[i])

        else:
         data[i] = data[i].fillna(-999)

    res = pd.concat(other_frames, axis=1)
    return res

def find_denominator(df, col):
    """
    Function that trying to find an approximate denominator used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 


def parse_data(intrain, intest):
  train, test = pd.read_csv(intrain), pd.read_csv(outtrain)
  for col in test.columns:
    if train[col.dtype] == "object":
        train[col]

def save_arch(mat, filename, type="train"):
 if type == "test":
    np.savez(filename, features = mat[:,1:], ids=mat[:,0])
 else:
    print "assuming train data"
    np.savez(filename, features=mat[:,2:], labels=mat[:,1])

intrain, intest, outtrain, outtest = argv[1:5]

train_data, test_data = parse_data(intrain, intest)

print train_data.shape, test_data.shape
save_arch(test_data.as_matrix(), outtest, type="test")
save_arch(train_data.as_matrix(), outtrain, type="train")


#td = d.as_matrix()
#np.savez("train1.npz", features = td[:,2:], labels=td[:,1])
#model = train_model(d)
#test_data = read_clean_csv('test.csv').as_matrix()

#res = model.predict_proba(test_data[:,1:])
#res[:,0] = test_data[:,0] #ids
#np.savetxt("sub2.csv", res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
