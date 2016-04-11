from utils import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, StratifiedKFold


train_features, train_labels, test_features, ids, outfile = read_data()

extc = ExtraTreesClassifier(n_estimators=1000,max_features= 50,criterion= 'entropy',min_samples_split= 4,
 max_depth= 40, min_samples_leaf= 2, n_jobs = -1)      

probs = cross_val_model(extc, train_features, train_labels, test_features)
save_submission(outfile, ids=ids, probs=probs)
print "saved model, validating now"
print cross_val_score(model, train_features, train_labels, scoring= "log_loss")
