from utils import read_data, save_submission
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from sys import argv, exit


#this file is an attempt to throw everything sklearn has to create a whole bunch of submissions which will then be averaged out.

train_features, train_labels, test_features, ids, outfile = read_data()

def train_et():
 et = ExtraTreesClassifier(n_estimators = 500, max_depth = 35, min_samples_split=4, min_samples_leaf=2, criterion="entropy")
 et.fit(train_features, train_labels)
 probs = et.predict_proba(test_features)[:,1]
 save_submission(outfile+"_et", ids, probs)
# print cross_val_score(et, train_features, train_labels, scoring="log_loss")

def train_rf():
 rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_split=4, min_samples_leaf=2, criterion="entropy")
 rf.fit(train_features, train_labels)
 probs = rf.predict_proba(test_features)[:,1]
 save_submission(outfile+"_rf", ids, probs)
 print cross_val_score(rf, train_features, train_labels, scoring="log_loss")

def train_ada():
    ada = AdaBoostClassifier(n_estimators=100)
    ada.fit(train_features, train_labels)
    probs = ada.predict_proba(test_features)[:,1]
    save_submission(outfile+"_ada", ids, probs)
#    print cross_val_score(ada, train_features, train_labels, scoring="log_loss")

def train_gb():
    gb = GradientBoostingClassifier(n_estimators=100)
    gb.fit(train_features, train_labels)
    probs = gb.predict_proba(test_features)[:,1]
    save_submission(outfile+"_gb", ids, probs)
    print "created submission for gb"
    print cross_val_score(gb, train_features, train_labels, scoring="log_loss")

train_et()
#print "Trained ET"
#train_rf()
#train_ada()
#print "Trained ADA"
#train_gb()
#print "Trained GB"
