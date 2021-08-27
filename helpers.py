import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(cuda.Device(0).name())
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

noPre = lambda x: x #place holder for no preprocessing

def loadData(frac=1, n_signal=-1):
  if n_signal > -1:
    frac = n_signal / 7e6

  print("Loading train data")
  with open("/vols/cms/mdk16/ggtt/ParamNN/data/all_train.pkl", "rb") as f:
    train_df = pickle.load(f).sample(frac=1).reset_index(drop=True)
  print("Loading test data")
  with open("/vols/cms/mdk16/ggtt/ParamNN/data/all_test.pkl", "rb") as f:
    test_df = pickle.load(f).sample(frac=1).reset_index(drop=True)

  train_df.loc[abs(train_df.mass-500)<1, "mass"] = 500.0
  test_df.loc[abs(train_df.mass-500)<1, "mass"] = 500.0

  return train_df, test_df

class TorchHelper:
  def __init__(self, preX=noPre):
    self.preX = preX

  def getBatches(self, X, y, batch_size, shuffle=False):
    if shuffle:
      shuffle_ids = np.random.permutation(len(X))
      X = X[shuffle_ids].copy()
      y = y[shuffle_ids].copy()
    for i_picture in range(0, len(X), batch_size):
      batch_X = X[i_picture:i_picture + batch_size]
      batch_y = y[i_picture:i_picture + batch_size]
    
    X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
    Y_torch = torch.tensor(batch_y, dtype=torch.float).to(dev)

    yield X_torch, Y_torch

  def getTotAvgLoss(self, model, loss_function, X, y, batch_size):
    losses = []
    for batch_X, batch_y in self.getBatches(X, y, batch_size):
      loss = loss_function(model(batch_X), batch_y)
      losses.append(loss.item())
    return sum(losses)/len(losses)

  def train(self, model, X_train, y_train, X_test, y_test, n_epochs=1, batch_size=32, lr=0.001):
    X_train = self.preX(X_train).to_numpy()
    X_test = self.preX(X_test).to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    test_loss = []

    for i_epoch in tqdm(range(n_epochs)):      
      for batch_X, batch_y in self.getBatches(X_train, y_train, batch_size, shuffle=True):
        loss = loss_function(model(batch_X), batch_y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

      train_loss.append(self.getTotAvgLoss(model, loss_function, X_train, y_train, batch_size))
      test_loss.append(self.getTotAvgLoss(model, loss_function, X_test, y_test, batch_size))
      
    return train_loss, test_loss

  def predict(self, model, X, batch_size=32):
    X = self.preX(X).to_numpy()
    predictions = []
    for i_picture in range(0, len(X), batch_size):
      batch_X = X[i_picture:i_picture + batch_size]
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      predictions.append(model(X_torch).to('cpu').detach().numpy())
    return np.concatenate(predictions)

  def getROC(self, model, X, y):
    predictions = self.predict(model, X)
    fpr, tpr, t = roc_curve(y, predictions)
    auc = roc_auc_score(y, predictions)
    return fpr, tpr, auc