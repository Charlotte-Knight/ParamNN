import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

all_features =  [
  'gg_mass', 'nJet', 'MET_pt', 'MET_phi', 'diphoton_pt_mgg',
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
  'Category_pairsLoose'
  ]

def minMaxNormFeatures(df, features):
  dff = df[features]
  df.loc[:, features] = (dff-dff.min())/(dff.max()-dff.min())
  return df

def stdMeanNormFeatures(df, features):
  dff = df[features]
  df.loc[:, features] = (dff-dff.mean())/dff.std()
  return df

def oneHot(df, features):
  new_columns = [df]
  for feature in features:
    new_columns.append(pd.get_dummies(df[feature], prefix=feature))
    df.drop(feature, axis=1, inplace=True)
  return pd.concat(new_columns, axis=1)
    
def normaliseFeatures(df):
  id_features = ['tau1_id_vs_e', 'tau1_id_vs_m', 'tau1_id_vs_j', 'tau2_id_vs_e', 'tau2_id_vs_m', 'tau2_id_vs_j', 'jet1_id', 'jet2_id']
  #print(id_features)
  #df = oneHot(df, id_features)

  other_features = list(filter(lambda x: x not in id_features, all_features))
  other_features.remove("gg_mass")

  df = minMaxNormFeatures(df, other_features)
  #df.loc[:,"reco_mX"] /= 1000
  return df

def addMX(df):
  H1_px = df.pt_tautauSVFitLoose * np.cos(df.phi_tautauSVFitLoose)
  H1_py = df.pt_tautauSVFitLoose * np.sin(df.phi_tautauSVFitLoose)
  H1_pz = df.pt_tautauSVFitLoose * np.sinh(df.eta_tautauSVFitLoose)
  H1_P = df.pt_tautauSVFitLoose * np.cosh(df.eta_tautauSVFitLoose)
  H1_E = np.sqrt(H1_P**2 + 125**2)

  ph1_pt = df.lead_pho_ptmgg * df.gg_mass
  ph2_pt = df.sublead_pho_ptmgg * df.gg_mass

  ph1_px = ph1_pt * np.cos(df.lead_pho_phi)
  ph1_py = ph1_pt * np.sin(df.lead_pho_phi)
  ph1_pz = ph1_pt * np.sinh(df.lead_pho_eta)
  ph1_E = ph1_pt * np.cosh(df.lead_pho_eta)

  ph2_px = ph2_pt * np.cos(df.sublead_pho_phi)
  ph2_py = ph2_pt * np.sin(df.sublead_pho_phi)
  ph2_pz = ph2_pt * np.sinh(df.sublead_pho_eta)
  ph2_E = ph2_pt * np.cosh(df.sublead_pho_eta)

  HH_px = H1_px + ph1_px + ph2_px
  HH_py = H1_py + ph1_py + ph2_py
  HH_pz = H1_pz + ph1_pz + ph2_pz
  HH_E = H1_E + ph1_E + ph2_E

  HH_m = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)

  df["reco_mX"] = HH_m
  return df

def loadData(test_frac=0.5):
  print(">> Loading samples")
  #path = "/home/hep/mdk16/PhD/ggtt/df_convert/Pass1/"
  path = "Pass1/"
  
  #files = ["HHggtautau_ResonantPresel_1tau0lep_2018.pkl", "HHggtautau_ResonantPresel_1tau1lep_2018.pkl", "HHggtautau_ResonantPresel_2tau_2018.pkl"]
  files = ["HHggtautau_ResonantPresel_1tau0lep_2018.pkl"]
  #files = ["HHggtautau_ResonantPresel_1tau1lep_2018.pkl", "HHggtautau_ResonantPresel_2tau_2018.pkl"]
  
  dfs = []
  for filename in files:
    print("> %s"%filename)
    with open(path+filename, "rb") as f:
      dfs.append(pickle.load(f))
  df = pd.concat(dfs, ignore_index=True)
  
  #df = addMX(df)

  df = normaliseFeatures(df)

  #tag signal and background
  df["y"] = 0
  df.loc[df.process_id<0, "y"] = 1

  #set masses
  df["mass"] = 0
  df.loc[df.process_id==-1, "mass"] = 0.300
  df.loc[df.process_id==-2, "mass"] = 0.400
  df.loc[df.process_id==-3, "mass"] = 0.500
  df.loc[df.process_id==-4, "mass"] = 0.800
  df.loc[df.process_id==-5, "mass"] = 1.000

  print("> Loaded %dk signal samples"%(sum(df.process_id<0)/1000))
  print("> Loaded %dk bkg samples"%(sum(df.process_id>0)/1000))
  print("> Loaded %dk data events"%(sum(df.process_id==0)/1000))

  print("> Sumw signal: %.2f"%df[df.process_id<0].weight.sum())
  print("> Sumw bkg: %.2f"%df[df.process_id>0].weight.sum())
  print("> Sumw data: %.2f"%df[df.process_id==0].weight.sum())

  MC = df[df.process_id!=0]
  data = df[df.process_id==0]

  print("> Test sample fraction = %.2f"%test_frac)
  train_df, test_df = train_test_split(MC, test_size=test_frac, random_state=1)  

  return train_df, test_df, data

def getXyw(train_df, test_df, masses, features):
  s = (train_df.y==0) | ((train_df.y==1)&(train_df.mass.isin(masses)))
  X_train = train_df[s][features]
  y_train = train_df[s]["y"]
  w_train = train_df[s]["weight"]

  s = (test_df.y==0) | ((test_df.y==1)&(test_df.mass.isin(masses)))
  X_test = test_df[s][features]
  y_test = test_df[s]["y"]
  w_test = test_df[s]["weight"]
  return X_train, y_train, X_test, y_test, w_train, w_test

def assignMassToBkg(df, mass):
  df.loc[df.y==0, "mass"] = mass
  return df

def plotRecoMX(df):
  for m in np.unique(df.mass):
    print(m)
    plt.hist(df[(df.y==1)&(df.mass==m)].reco_mX, bins=100, range=(0, 1.2), label="%d"%int(m*1000), density=True)
  plt.legend()
  plt.savefig("mX.png")

if __name__=="__main__":
  train_df, test_df, data_df = loadData(0.5)
  print("> Training samples")
  print(train_df.head(10))
  print("\n> Test samples")
  print(test_df.head(10))
  print("\n> Data")
  print(data_df.head(10))

  #plotRecoMX(train_df)
