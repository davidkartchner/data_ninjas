from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.optimizers import SGD, Adagrad
from sys import argv, exit

if len(argc) < 3:
    print "Usage training_npz submission_file"
    exit()

infile = argc[1]
out = argv[2]

archive = np.load(infile)
X = archive['features']
y = archive['labels']
num_neurons = 10

model = Sequential()
model.add(Dense(num_neurons, input_dim = X.shape[1]))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(optimizer = 'sgd', loss = 'cross-entropy')
