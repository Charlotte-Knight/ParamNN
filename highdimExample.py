import helpers
import torch
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
print(dev)

def preX(X):
  X_copy = X.copy()
  X_copy.loc[:, "mass"] /= 1000
  return X_copy

"""-------------------------------------------------------------------------------"""

train_df, test_df = helpers.loadData(frac=0.1)

important_features = ["f%d"%i for i in [0,3,6,10,14,23,26]]
#important_features = ["f26"]
important_features.append("mass")
n_features = len(important_features)

all_masses = [500, 750, 1000, 1250, 1500]
train_masses = [750, 1000, 1250, 1500]

X_train = train_df[train_df.mass.isin(train_masses)][important_features]
X_test = test_df[test_df.mass.isin(train_masses)][important_features]
y_train = train_df[train_df.mass.isin(train_masses)]["y"]
y_test = test_df[test_df.mass.isin(train_masses)]["y"]

"""-------------------------------------------------------------------------------"""

th = helpers.TorchHelper(preX)

model = torch.nn.Sequential(
  torch.nn.Linear(n_features,10),
  torch.nn.ELU(),
  torch.nn.Linear(10,10),
  torch.nn.ELU(),
  torch.nn.Linear(10,1),
  torch.nn.Flatten(0,1),
  torch.nn.Sigmoid()
)

model.to(dev)

train_loss, test_loss = th.train(model, X_train, y_train, X_test, y_test, n_epochs=1, batch_size=128, lr=0.01)

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

#put interp masses back in
X_train = train_df[important_features]
X_test = test_df[important_features]
y_train = train_df["y"]
y_test = test_df["y"]

for mass in all_masses:
  #output score on train set
  bkg_scores = th.predict(model, X_train[(X_train.mass==mass)&(y_train==0)])
  sig_scores = th.predict(model, X_train[(X_train.mass==mass)&(y_train==1)])

  plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5)
  plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5)
  plt.legend()
  plt.yscale("log")
  plt.savefig("plots/highdimExample/train_scores_m%d.png"%mass)
  plt.clf()

  #output score on train set
  bkg_scores = th.predict(model, X_test[(X_test.mass==mass)&(y_test==0)])
  sig_scores = th.predict(model, X_test[(X_test.mass==mass)&(y_test==1)])

  plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5)
  plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5)
  plt.legend()
  plt.yscale("log")
  plt.savefig("plots/highdimExample/test_scores_m%d.png"%mass)
  plt.clf()
  
  #ROC curves
  fpr, tpr, score = th.getROC(model, X_train[X_train.mass==mass], y_train[X_train.mass==mass])
  plt.plot(fpr, tpr, label="Train AUC=%.4f"%score)
  fpr, tpr, score = th.getROC(model, X_test[X_test.mass==mass], y_test[X_test.mass==mass])
  plt.plot(fpr, tpr, label="Test AUC=%.4f"%score)
  plt.legend()
  plt.savefig("plots/highdimExample/ROC_m%d.png"%mass)
  plt.clf()
