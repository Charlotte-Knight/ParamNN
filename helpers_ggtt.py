import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

noPre = lambda x: x #place holder for no preprocessing

def normaliseFeatures(df):
  features = ['gg_mass', 'nJet', 'MET_pt', 'MET_phi', 'diphoton_pt_mgg',
       'diphoton_rapidity', 'diphoton_delta_R', 'lead_pho_ptmgg',
       'sublead_pho_ptmgg', 'lead_pho_eta', 'sublead_pho_eta',
       'lead_pho_idmva', 'sublead_pho_idmva', 'lead_pho_phi',
       'sublead_pho_phi', 'ele1_pt', 'ele1_eta', 'ele1_phi', 'ele2_pt',
       'ele2_eta', 'ele2_phi', 'ele1_tightId', 'ele2_tightId', 'muon1_pt',
       'muon1_eta', 'muon1_phi', 'muon2_pt', 'muon2_eta', 'muon2_phi',
       'muon1_tightId', 'muon2_tightId', 'tau1_pt', 'tau1_eta', 'tau1_phi',
       'tau2_pt', 'tau2_eta', 'tau2_phi', 'tau1_id_vs_e', 'tau1_id_vs_m',
       'tau1_id_vs_j', 'tau2_id_vs_e', 'tau2_id_vs_m', 'tau2_id_vs_j', 'n_tau',
       'n_electrons', 'n_muons', 'jet1_pt', 'jet1_eta', 'jet1_id',
       'jet1_bTagDeepFlavB', 'jet2_pt', 'jet2_eta', 'jet2_id',
       'jet2_bTagDeepFlavB', 'pt_tautauSVFitLoose', 'eta_tautauSVFitLoose',
       'phi_tautauSVFitLoose', 'm_tautauSVFitLoose', 'dR_tautauSVFitLoose',
       'dR_ggtautauSVFitLoose', 'dphi_MET_tau1', 'dR_tautauLoose',
       'dR_ggtautauLoose', 'dPhi_ggtautauLoose', 'dPhi_ggtautauSVFitLoose',
       'Category_pairsLoose']
  dff = df[features]
  df.loc[:, features] = (dff-dff.min())/(dff.max()-dff.min())
  return df

def loadData(frac=1):
  print("Loading data")
  dfs = []
  #path = "/home/hep/mdk16/PhD/ggtt/df_convert/Pass1/"
  path = "Pass1/"
  #files = ["HHggtautau_ResonantPresel_1tau0lep_2018.pkl", "HHggtautau_ResonantPresel_1tau1lep_2018.pkl", "HHggtautau_ResonantPresel_2tau_2018.pkl"]
  files = ["HHggtautau_ResonantPresel_1tau0lep_2018.pkl"]
  for filename in files:
    with open(path+filename, "rb") as f:
      dfs.append(pickle.load(f))
  df = pd.concat(dfs, ignore_index=True)
  
  df = normaliseFeatures(df)

  #tag signal and background
  df["y"] = 0
  df.loc[df.process_id<0, "y"] = 1

  #set masses
  df["mass"] = 0
  df.loc[df.process_id==-1, "mass"] = 300
  df.loc[df.process_id==-2, "mass"] = 400
  df.loc[df.process_id==-3, "mass"] = 500
  df.loc[df.process_id==-4, "mass"] = 800
  df.loc[df.process_id==-5, "mass"] = 1000

  #df = df[(df.process_id<=0)|(df.process_id==3)]

  print("Loaded %dk signal samples"%(sum(df.process_id<0)/1000))
  print("Loaded %dk bkg samples"%(sum(df.process_id>0)/1000))
  print("Loaded %dk data events"%(sum(df.process_id==0)/1000))

  MC = df[df.process_id!=0]
  data = df[df.process_id==0]

  train_df, test_df = train_test_split(MC, test_size=0.5, random_state=1)  

  return train_df, test_df, data

class TorchHelper:
  def __init__(self, preX=noPre):
    self.preX = preX

  def getBatches(self, X, y, w, batch_size, shuffle=False):
    if shuffle:
      shuffle_ids = np.random.permutation(len(X))
      X_sh = X[shuffle_ids].copy()
      y_sh = y[shuffle_ids].copy()
      w_sh = w[shuffle_ids].copy()
    else:
      X_sh = X.copy()
      y_sh = y.copy()
      w_sh = w.copy()
    for i_picture in range(0, len(X), batch_size):
      batch_X = X_sh[i_picture:i_picture + batch_size]
      batch_y = y_sh[i_picture:i_picture + batch_size]
      batch_w = w_sh[i_picture:i_picture + batch_size]
    
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      y_torch = torch.tensor(batch_y, dtype=torch.float).to(dev)
      w_torch = torch.tensor(batch_w, dtype=torch.float).to(dev)

      yield X_torch, y_torch, w_torch

  def MSELoss(self, input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

  def BCELoss(self, input, target, weight):
    x, y, w = input, target, weight
    #log = lambda x: torch.clamp(torch.log(x), min=-100, max=100)
    log = lambda x: torch.log(x+1e-8)
    return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))

  def getTotAvgLoss(self, model, loss_function, X, y, w, batch_size):
    losses = []
    for batch_X, batch_y, batch_w in self.getBatches(X, y, w, batch_size):
      loss = loss_function(model(batch_X), batch_y, batch_w)
      losses.append(loss.item())
    return sum(losses)/len(losses)
    #return np.median(losses)

  def equaliseWeights(self, w, y):
    w[y==1] = w[y==1] * 10000
    w[y==0] = w[y==0] * (sum(w[y==1]) / sum(w[y==0]))
    return w

  def cullSignal(self, X, y, w):
    #keep as much signal as there is background
    nsig = sum(y==1)
    nbkg = sum(y==0)
    signal_indices = np.arange(nsig+nbkg)[y==1]
    bkg_indices = np.arange(nsig+nbkg)[y==0]
    keep_signal_indices = np.random.choice(signal_indices, nbkg, replace=False)
    
    selection = np.concatenate([bkg_indices, keep_signal_indices])
    return X[selection], y[selection], w[selection]

  def printSampleSummary(self, X_train, y_train, X_test, y_test, w_train, w_test):
    print("Training set:")
    print(" nsig = %d"%sum(y_train==1))
    print(" nbkg = %d"%sum(y_train==0))
    print(" sum wsig = %f"%sum(w_train[y_train==1]))
    print(" sum wbkg = %f"%sum(w_train[y_train==0]))

    print("Test set:")
    print(" nsig = %d"%sum(y_test==1))
    print(" nbkg = %d"%sum(y_test==0))
    print(" sum wsig = %f"%sum(w_test[y_test==1]))
    print(" sum wbkg = %f"%sum(w_test[y_test==0]))

  def shouldEarlyStop(self, train_loss):
    """
    Want to stop if seeing no appreciable improvment.
    Check 1. Was the best score more than y epochs ago?
          2. If score is improving, has it improved by more than
             x percent over y epochs. 
    """
    x = 0.01
    y = 10

    n_epochs = len(train_loss)

    if n_epochs < y*2:
      return False

    train_loss = np.array(train_loss)
    train_loss = train_loss.sum(axis=1)

    best_loss = train_loss.min()
    best_loss_epoch = np.where(train_loss==best_loss)[0][0] + 1

    if n_epochs - best_loss_epoch > y:
      print("Best loss happened a while ago")
      return True

    best_loss_before = train_loss[:-y].min()
    if (best_loss_before-best_loss)/best_loss < x:
      print("Not enough improvement")
      return True

    return False    

  def train(self, model, X_train, y_train, X_test, y_test, w_train, w_test, n_epochs=1, batch_size=32, lr=0.001, wd=0):
    X_train = self.preX(X_train).to_numpy()
    X_test = self.preX(X_test).to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    w_train = w_train.to_numpy()
    w_test = w_test.to_numpy()

    X_train, y_train, w_train = self.cullSignal(X_train, y_train, w_train)
    X_test, y_test, w_test = self.cullSignal(X_test, y_test, w_test)

    w_train = self.equaliseWeights(w_train, y_train)
    w_test = self.equaliseWeights(w_test, y_test)
    
    self.printSampleSummary(X_train, y_train, X_test, y_test, w_train, w_test)

    train_masses = sorted(np.unique(X_train[:,-1]))

    #loss_function = torch.nn.MSELoss()
    #loss_function = self.MSELoss
    loss_function = self.BCELoss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_loss = []
    test_loss = []

    for i_epoch in tqdm(range(n_epochs)):     
      for batch_X, batch_y, batch_w in tqdm(self.getBatches(X_train, y_train, w_train, batch_size, shuffle=True), leave=False):
        loss = loss_function(model(batch_X), batch_y, batch_w)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
      scheduler.step()

      trl = []
      tel = []
      for mass in train_masses:
        trl.append(self.getTotAvgLoss(model, loss_function, X_train[X_train[:,-1]==mass], y_train[X_train[:,-1]==mass], w_train[X_train[:,-1]==mass], batch_size))
        tel.append(self.getTotAvgLoss(model, loss_function, X_test[X_test[:,-1]==mass], y_test[X_test[:,-1]==mass], w_test[X_test[:,-1]==mass], batch_size))
      train_loss.append(trl)
      test_loss.append(tel)
      
      if self.shouldEarlyStop(train_loss):
        break

      #shuffle masses in bkg
      X_train[y_train==0,-1] = np.random.choice(train_masses, len(X_train[y_train==0][:,-1]))
      X_test[y_test==0,-1] = np.random.choice(train_masses, len(X_test[y_test==0][:,-1]))
      
    return train_loss, test_loss

  def predict(self, model, X, batch_size=32):
    X = self.preX(X).to_numpy()
    predictions = []
    for i_picture in range(0, len(X), batch_size):
      batch_X = X[i_picture:i_picture + batch_size]
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      predictions.append(model(X_torch).to('cpu').detach().numpy())
    return np.concatenate(predictions)

  def getROC(self, model, X, y, weight=None):
    predictions = self.predict(model, X)
    fpr, tpr, t = roc_curve(y, predictions, sample_weight=weight)
    #fpr, tpr, t = roc_curve(y, predictions)
    try:
      auc = roc_auc_score(y, predictions, sample_weight=weight)
      #auc = roc_auc_score(y, predictions)
    except:
      auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

  # def getROC(self, model, X, y, weight=None):
  #   predictions = self.predict(model, X)
  #   predictions = pd.DataFrame({"pred":predictions, "y":y})
  #   predictions.sort_values(["pred"], inplace=True)
  #   print(predictions)
  #   fpr = []
  #   tpr = []
  #   sig_count = 0
  #   bkg_count = 0
  #   tot_sig = sum(predictions.y==1)
  #   tot_bkg = sum(predictions.y==0)
  #   for i in range(len(predictions)):
  #     if predictions.iloc[i].y == 0:
  #       bkg_count += 1
  #     else:
  #       sig_count += 1
  #     fpr.append((tot_bkg-bkg_count)/tot_bkg)
  #     tpr.append((tot_sig-sig_count)/tot_sig)
    
  #   auc = np.trapz(tpr, fpr)
  #   return fpr, tpr, auc
