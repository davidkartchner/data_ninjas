from utils import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import cross_val_score, StratifiedKFold


train_features, train_labels, test_features, ids, outfile = read_data()

extc = ExtraTreesClassifier(n_estimators=1200,max_features= 50,criterion= 'entropy',min_samples_split= 4,
 max_depth= 35, min_samples_leaf= 2, n_jobs = -1)

num_reps = 5
tot = np.zeros(len(test_features))
for i in xrange(num_reps):
    probs, tprobs = cross_val_model(extc, train_features, train_labels, test_features, return_trainprobs=True)
    np.savez(outfile+"_feature"+str(i), test=probs, train=tprobs)
    tot += probs
tot/=num_reps
save_submission(outfile, ids=ids, probs=tot)
print "saved model, validating now"
#print cross_val_score(model, train_features, train_labels, scoring= "log_loss")

