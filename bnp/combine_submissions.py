from sys import argv, exit
import pandas as pd

ids = None
tot = len(argv)-2

if tot <= 1:
    print "Must combine more than one submission"

probs = None
for filename in argv[1:-1]:
  df = pd.read_csv(filename)
  print filename, df
  if probs is None:
      probs = df["PredictedProb"].as_matrix()
  else:
      probs += df["PredictedProb"].as_matrix()
  ids = df["ID"]

probs /= float(tot)
print probs
pd.DataFrame({"ID":ids, "PredictedProb":probs}).to_csv(argv[-1], index= False)
