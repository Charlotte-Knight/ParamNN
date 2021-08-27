import helpers
import torch
import matplotlib.pyplot as plt

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

def preX(X):
  X_copy = X.copy()
  X_copy.loc[:, "mass"] /= 1000
  return X_copy

"""-------------------------------------------------------------------------------"""

train_df, test_df = helpers.loadData(n_signal=50000)

important_features = ["f%d"%i for i in [0,3,6,10,14,23,26]]
#important_features = ["f26"]
important_features.append("mass")
n_features = len(important_features)

train_masses = [1250,1500]
train_masses = [750, 1250, 1500]
interp_masses = [1000]

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

train_loss, test_loss = th.train(model, X_train, y_train, X_test, y_test, n_epochs=50, batch_size=32, lr=0.025)

"""-------------------------------------------------------------------------------"""

#loss plot
plt.plot(test_loss, label="test")
plt.plot(train_loss, label="train")
plt.legend()
plt.savefig("plots/highdimExample/loss.png")
plt.clf()

for mass in train_masses:
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