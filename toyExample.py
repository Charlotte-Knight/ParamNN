import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def preprocess_X(X):
  X_preprocessed = torch.tensor(X, dtype=torch.float).reshape(-1, 2)
  return X_preprocessed

def preprocess_y(y):
  y_preprocessed = torch.tensor(y, dtype=torch.float)
  return y_preprocessed

def get_batches(X, y, batch_size, shuffle=False):
  if shuffle:
    shuffle_ids = np.random.permutation(len(X))
    X = X[shuffle_ids].copy()
    y = y[shuffle_ids].copy()
  for i_picture in range(0, len(X), batch_size):
    batch_X = X[i_picture:i_picture + batch_size]
    batch_y = y[i_picture:i_picture + batch_size]
    
    yield preprocess_X(batch_X), preprocess_y(batch_y)

def pred(model, X_batch):
  return model(X_batch).to('cpu').detach().numpy()

def get_predictions(model, X):
  preX = preprocess_X(X)
  preds = np.concatenate([pred(model, preX[i:i+batch_size]) for i in range(0, len(X), batch_size)], axis=0)
  return preds

def getSignalSamples(theta, sigma=0.1):
  return np.random.normal(theta, sigma, N)

def getBackgroundSamples():
  return np.random.uniform(-5, 5, N)

try:
  os.makedirs("plots/toyExample")
except:
  pass

"""-------------------------------------------------------------------------------"""

N = 10000
train_masses = [-2,0,2]
interp_masses = [-1,1]

Xs = []
ys = []
for mass in train_masses:
  #signal
  reco = getSignalSamples(mass)[:,np.newaxis]
  truth = (np.ones(N)*mass)[:,np.newaxis]
  Xs.append(np.concatenate([truth, reco], axis=1))
  ys.append(np.ones(N))
  
  #background
  reco = getBackgroundSamples()[:,np.newaxis]
  truth = (np.ones(N)*mass)[:,np.newaxis]
  Xs.append(np.concatenate([truth, reco], axis=1))
  ys.append(np.zeros(N))

X = np.concatenate(Xs, axis=0)
y = np.concatenate(ys, axis=0)

plt.hist(X[y==0][:,1], bins=100, range=(-5,5), alpha=0.5, label="bkg")
for mass in train_masses:
  plt.hist(X[(y==1)&(X[:,0]==mass)][:,1], bins=100, range=(-5,5), alpha=0.5, label=r"$\theta$=%d"%mass)
plt.xlabel("Reco Mass")
plt.legend()
plt.savefig("plots/toyExample/simple_gauss_train.png")
plt.clf()

"""-------------------------------------------------------------------------------"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=39)
  
loss_function = torch.nn.MSELoss()

model = torch.nn.Sequential(
  torch.nn.Linear(2,10),
  torch.nn.ELU(),
  torch.nn.Linear(10,10),
  torch.nn.ELU(),
  torch.nn.Linear(10,1),
  torch.nn.Flatten(0,1),
  torch.nn.Sigmoid()
)
 
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 10
batch_size = 100

train_loss = []
test_loss = []

for i_epoch in range(n_epochs):
  for batch_X, batch_y in get_batches(X_test, y_test, batch_size=batch_size):
    loss = loss_function(model(batch_X), batch_y)
    test_loss.append(loss.item())

  print(i_epoch)
  for batch_X, batch_y in get_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
    loss = loss_function(model(batch_X), batch_y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

"""-------------------------------------------------------------------------------"""

bkg = X[y==0]
sig = X[y==1]

bkg_scores = get_predictions(model, bkg)
sig_scores = get_predictions(model, sig)

plt.hist(bkg_scores, bins=100, range=(0,1), label="bkg", alpha=0.5)
plt.hist(sig_scores, bins=100, range=(0,1), label="sig", alpha=0.5)
#plt.hist(bkg_scores, bins=100, label="bkg", alpha=0.5)
#plt.hist(sig_scores, bins=100, label="sig", alpha=0.5)
plt.legend()
plt.yscale("log")
plt.savefig("plots/toyExample/scores.png")
plt.clf()

plt.plot(test_loss)
plt.savefig("plots/toyExample/loss.png")
plt.clf()

"""-------------------------------------------------------------------------------"""

for i, mass in enumerate(train_masses):
  reco = np.linspace(-5,5,1000)[:,np.newaxis]
  truth = (np.ones(1000)*mass)[:,np.newaxis]
  test_X = np.concatenate([truth,reco], axis=1)

  scores = get_predictions(model, test_X)
  if i==0: plt.plot(reco, scores, "k", label="trained")
  else: plt.plot(reco, scores, "k")

for i, mass in enumerate(interp_masses):
  reco = np.linspace(-5,5,1000)[:,np.newaxis]
  truth = (np.ones(1000)*mass)[:,np.newaxis]
  test_X = np.concatenate([truth,reco], axis=1)

  scores = get_predictions(model, test_X)
  if i==0: plt.plot(reco, scores, "r--", label="interpolated")
  else: plt.plot(reco, scores, "r--")

plt.legend()
plt.xlabel("Reco Mass")
plt.ylabel("Output score")
plt.savefig("plots/toyExample/interp.png")
plt.clf()

