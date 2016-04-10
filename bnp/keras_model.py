import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from keras.regularizers import l2, activity_l2
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.optimizers import SGD, Adagrad
from sys import argv, exit

if len(argv) < 4:
    print "Usage: training file test file output file"
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

#assert len(ids) == len(test_features)
print "loaded data"
res = np.zeros((len(ids),2))

num_neurons = 30
model = Sequential()
model.add(Dense(num_neurons, input_dim = train_features.shape[1], W_regularizer = l2(.01), activity_regularizer = activity_l2(.01)))
model.add(Activation('sigmoid'))

for i in xrange(3):
 model.add(Dense(num_neurons, input_dim = num_neurons, W_regularizer = l2(.01), activity_regularizer = activity_l2(.01)))
 # model.add(Dropout(.5))
 model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'sgd', loss = 'binary_crossentropy')

model.fit(train_features, train_labels, validation_split = .1, verbose = 1, nb_epoch = 20)
print "Trained"
tpreds = np.ravel(model.predict(train_features))
print np.log(tpreds).dot(train_labels) / len(ids)

#print np.ravel(model.predict(test_features))
res[:,1] = np.ravel(model.predict(test_features))
res[:,0] = ids #ids
np.savetxt(outfile, res, fmt = ["%d","%f"] ,header = "ID,PredictedProb",comments = "", delimiter=",")
