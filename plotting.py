import matplotlib.pyplot as plt
import numpy as np
import os

def makeWayForSave(path):
  savedir = "/".join(path.split("/")[:-1])
  try:
    os.makedirs(savedir)
  except:
    pass


def plotOutputScore(model, X, y, w, saveas):
  bkg_scores = model.predict(X[y==0])
  sig_scores = model.predict(X[y==1])

  bkg_w = w[y==0]
  sig_w = w[y==1]

  plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5, weights=bkg_w)
  plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5, weights=sig_w)
  plt.legend()
  plt.yscale("log")
  makeWayForSave(saveas)
  plt.savefig(saveas)
  plt.clf()

def plotLoss(train_losses, test_losses, saveas):
  plt.plot(train_losses, label="train")
  plt.plot(test_losses, label="test")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.legend()
  makeWayForSave(saveas)
  plt.savefig(saveas)
  plt.clf()

def plotAUC(model, X_train, y_train, X_test, y_test, w_train, w_test, saveas):
  fpr, tpr, train_score = model.getROC(X_train, y_train, w_train)
  plt.plot(fpr, tpr, label="Train AUC=%.4f"%train_score)
  fpr, tpr, test_score = model.getROC(X_test, y_test, w_test)
  plt.plot(fpr, tpr, label="Test AUC=%.4f"%test_score)

  plt.xscale("log")
  plt.legend()
  makeWayForSave(saveas)
  plt.savefig(saveas)
  plt.clf()

  return train_score, test_score

def plotDistributions(model, X, y, w, saveinto):
  scores = model.predict(X)

  bkg_scores = scores[y==0]
  sig_scores = scores[y==1]

  bkg_w = w[y==0]
  sig_w = w[y==1]

  for feature in X.columns:
    if feature != "mass":
      fig, ax1 = plt.subplots()

      ax1.hist(X[y==0][feature], bins=20, range=(0,1), label="bkg", alpha=0.5, weights=bkg_w)
      ax1.hist(X[y==1][feature], bins=20, range=(0,1), label="sig", alpha=0.5, weights=sig_w)
      ax1.set_yscale('log')
      ax1.set_ylabel("Sum of weights")
      ax1.set_xlabel(feature)
      ax1.legend()

      ax2 = ax1.twinx()

      n, bin_edges = np.histogram(X[feature], bins=20, range=(0,1), weights=w)
      tot_score, bin_edges = np.histogram(X[feature], bins=20, range=(0,1), weights=(scores * w))
      with np.errstate(divide='ignore',invalid='ignore'):
        ax2.plot((bin_edges[:-1]+bin_edges[1:])/2, tot_score/n, label="NN Output")
        ax2.scatter((bin_edges[:-1]+bin_edges[1:])/2, tot_score/n, label="NN Output")
      ax2.set_ylim(-0.1,1.1)
      ax2.set_ylabel("Average output score")
      ax2.legend()
      
      saveas = "%s/%s.png"%(saveinto, feature)
      makeWayForSave(saveas)
      fig.savefig(saveas)
      plt.close()