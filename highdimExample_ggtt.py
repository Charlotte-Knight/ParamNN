import helpers_ggtt as helpers
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

seed = 2
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
print(dev)

def preX(X):
  X_copy = X.copy()
  X_copy.loc[:, "mass"] /= 1000
  return X_copy

def assignRandomMass(df, masses):
  n_bkg = sum(df.y==0)
  df.loc[df.y==0, "mass"] = np.random.choice(masses, n_bkg)
  return df

def assignMass(df, mass):
  df.loc[df.y==0, "mass"] = mass
  return df

def getXyw(train_df, test_df, masses, features):
  X_train = train_df[train_df.mass.isin(masses)][features]
  X_test = test_df[test_df.mass.isin(masses)][features]
  y_train = train_df[train_df.mass.isin(masses)]["y"]
  y_test = test_df[test_df.mass.isin(masses)]["y"]
  w_train = train_df[train_df.mass.isin(masses)]["weight"]
  w_test = test_df[test_df.mass.isin(masses)]["weight"]
  return X_train, y_train, X_test, y_test, w_train, w_test


def getXywData(data, features):
  X_data = data[features]
  y_data = data["y"]
  w_data = data["weight"]
  return X_data, y_data, w_data

def replaceBkgWithData(X_test, y_test, w_test, X_data, y_data, w_data):
  X_test_new = pd.concat([X_test[y_test==1], X_data])
  y_test_new = pd.concat([y_test[y_test==1], y_data])
  w_test_new = pd.concat([w_test[y_test==1], w_data])
  return X_test_new, y_test_new, w_test_new

def equaliseSignalBkgWeights(df):
  #df.loc[:, "weight"] = 1

  masses = np.unique(df.mass)
  for m in masses:
    s = (df.y==1) & (df.mass==m)
    df.loc[s, "weight"] = df.loc[s, "weight"] * (1/sum(df.loc[s, "weight"]))
  
  df.loc[df.y==0, "weight"] = df.loc[df.y==0, "weight"] * (1/sum(df.loc[df.y==0, "weight"]))
  return df

def plotDistributions(train_df, features, mass):
  df = train_df[train_df.mass==mass]
  for feature in features:
    if feature != "mass":
      plt.hist(df[df.y==0][feature], bins=100, weights=df[df.y==0]["weight"], label="bkg", alpha=0.5)
      plt.hist(df[df.y==1][feature], bins=100, weights=df[df.y==1]["weight"], label="sig", alpha=0.5)
      plt.yscale("log")
      plt.legend()
      plt.savefig("plots/highdimExample/%s_%d.png"%(feature,mass))
      plt.clf()

"""-------------------------------------------------------------------------------"""

train_df, test_df, data = helpers.loadData()

#important_features = ["diphoton_pt_mgg", "diphoton_delta_R", "dR_tautauSVFitLoose", "tau1_id_vs_j"]
#important_features = ['diphoton_delta_R', 'tau1_pt', 'lead_pho_ptmgg', 'sublead_pho_ptmgg', 'diphoton_pt_mgg', 'dphi_MET_tau1']
important_features = ['diphoton_delta_R', 'tau1_pt']
important_features.append("mass")
n_features = len(important_features)

all_masses = [300, 400, 500, 800, 1000]
train_masses = [300]

train_df = assignRandomMass(train_df, train_masses)
test_df = assignRandomMass(test_df, train_masses)
#data = assignRandomMass(data, train_masses)

#train_df = equaliseSignalBkgWeights(train_df)
#test_df = equaliseSignalBkgWeights(test_df)

plotDistributions(train_df, important_features, train_masses[0])

X_train, y_train, X_test, y_test, w_train, w_test = getXyw(train_df, test_df, train_masses, important_features)

"""-------------------------------------------------------------------------------"""

th = helpers.TorchHelper(preX)

# model = torch.nn.Sequential(
#   torch.nn.Linear(n_features,10),
#   torch.nn.ELU(),
#   #torch.nn.Linear(10,10),
#   #torch.nn.ELU(),
#   torch.nn.Linear(10,1),
#   torch.nn.Flatten(0,1),
#   torch.nn.Sigmoid()
# )

model = torch.nn.Sequential(
  torch.nn.Linear(n_features,1),
  torch.nn.ELU(),
  #torch.nn.Linear(10,10),
  #torch.nn.ELU(),
  #torch.nn.Linear(10,1),
  torch.nn.Flatten(0,1),
  torch.nn.Sigmoid()
)

model.to(dev)

#train_loss, test_loss = th.train(model, X_train, y_train, X_test, y_test, w_train, w_test, n_epochs=1, batch_size=32, lr=0.001, wd=0.0)
train_loss, test_loss = th.train(model, X_train, y_train, X_test, y_test, w_train, w_test, n_epochs=5, batch_size=32, lr=0.005, wd=0.0)

"""-------------------------------------------------------------------------------"""

train_loss, test_loss = np.array(train_loss), np.array(test_loss)

#loss plots per mass
for i, mass in enumerate(train_masses):
  plt.plot(test_loss[:,i], label="test")
  plt.plot(train_loss[:,i], label="train")
  plt.legend()
  plt.savefig("plots/highdimExample/loss_%s.png"%mass)
  plt.clf()

#total loss plot
plt.plot(test_loss.sum(axis=1), label="test")
plt.plot(train_loss.sum(axis=1), label="train")
plt.legend()
plt.savefig("plots/highdimExample/loss.png")
plt.clf()

#th.printSampleSummary(X_train, y_train, X_test, y_test, w_train, w_test)

record = []

for mass in all_masses:
  print(mass)
  #output score on train set
  train_df = assignMass(train_df, mass)
  test_df = assignMass(test_df, mass)
  X_train, y_train, X_test, y_test, w_train, w_test = getXyw(train_df, test_df, [mass], important_features)

  bkg_scores = th.predict(model, X_train[(X_train.mass==mass)&(y_train==0)])
  sig_scores = th.predict(model, X_train[(X_train.mass==mass)&(y_train==1)])

  bkg_w = w_train[(X_train.mass==mass)&(y_train==0)]
  sig_w = w_train[(X_train.mass==mass)&(y_train==1)]

  plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5, weights=bkg_w)
  plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5, weights=sig_w)
  plt.legend()
  plt.yscale("log")
  plt.savefig("plots/highdimExample/train_scores_m%d.png"%mass)
  plt.clf()

  f = X_train[(X_train.mass==mass)].diphoton_delta_R
  scores = th.predict(model, X_train[(X_train.mass==mass)])
  # plt.hist(bkg_f, bins=100, label="bkg", alpha=0.5, weights=bkg_scores)
  # plt.hist(sig_f, bins=100, label="sig", alpha=0.5, weights=sig_scores)
  # plt.legend()
  # plt.yscale("log")
  # plt.savefig("plots/highdimExample/feature_scores_m%d.png"%mass)
  # plt.clf()
  n, bin_edges = np.histogram(f, bins=100, range=(0,1), weights=w_train[X_train.mass==mass])
  tot_score, bin_edges = np.histogram(f, bins=100, range=(0,1), weights=(scores * w_train[X_train.mass==mass]))
  plt.plot(bin_edges[:-1], tot_score/n)
  plt.ylim(-0.1,1.1)
  plt.savefig("plots/highdimExample/feature_scores_m%d.png"%mass)
  plt.clf()


  #output score on test set
  bkg_scores = th.predict(model, X_test[(X_test.mass==mass)&(y_test==0)])
  sig_scores = th.predict(model, X_test[(X_test.mass==mass)&(y_test==1)])

  bkg_w = w_test[(X_test.mass==mass)&(y_test==0)]
  sig_w = w_test[(X_test.mass==mass)&(y_test==1)]

  plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5, weights=bkg_w)
  plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5, weights=sig_w)
  plt.legend()
  plt.yscale("log")
  plt.savefig("plots/highdimExample/test_scores_m%d.png"%mass)
  plt.clf()
  
  #ROC curves
  fpr, tpr, train_score = th.getROC(model, X_train[X_train.mass==mass], y_train[X_train.mass==mass], w_train[X_train.mass==mass])
  plt.plot(fpr, tpr, label="Train AUC=%.4f"%train_score)
  fpr, tpr, test_score = th.getROC(model, X_test[X_test.mass==mass], y_test[X_test.mass==mass], w_test[X_test.mass==mass])
  plt.plot(fpr, tpr, label="Test AUC=%.4f"%test_score)

  data = assignMass(data, mass)
  X_data, y_data, w_data = getXywData(data, important_features)
  X_test_new, y_test_new, w_test_new = replaceBkgWithData(X_test, y_test, w_test, X_data, y_data, w_data)
  
  fpr, tpr, data_score = th.getROC(model, X_test_new[X_test_new.mass==mass], y_test_new[X_test_new.mass==mass], w_test_new[X_test_new.mass==mass])
  plt.plot(fpr, tpr, label="Data AUC=%.4f"%data_score)

  plt.xscale("log")
  plt.legend()
  plt.savefig("plots/highdimExample/ROC_m%d.png"%mass)
  plt.clf()

  record.append([mass, train_score, test_score, data_score])

with open("auc_record.txt", "a") as f:
  f.write("\nseed = %d\n"%seed)
  for each in record:
    f.write("%d %.4f %.4f %.4f\n"%(each[0], each[1], each[2], each[3]))

