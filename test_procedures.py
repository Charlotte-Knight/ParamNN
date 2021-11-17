import data_handling
#import paramNN
import models as paramNN
import plotting
import numpy as np
from collections import OrderedDict as od
import concurrent.futures
import pandas as pd

import limit

paramNN.setSeed(10)

def getLimit(model, sig_df, bkg_df, train_features):
  sig_scores = model.predict(sig_df[train_features])
  bkg_scores = model.predict(bkg_df[train_features])

  sig = pd.DataFrame({"mass": sig_df.gg_mass, "score": sig_scores, "weight": sig_df.weight})
  bkg = pd.DataFrame({"mass": bkg_df.gg_mass, "score": bkg_scores, "weight": bkg_df.weight})

  optimal_limit, optimal_boundary, boundaries, limits, ams = limit.optimiseBoundary(bkg, sig)
  return optimal_limit, optimal_boundary, boundaries[-1]


def plotAll(model, train_df, test_df, train_features, train_loss, test_loss, saveinto):
  train_loss, test_loss = np.array(train_loss), np.array(test_loss)
  plotting.plotLoss(train_loss.sum(axis=1), test_loss.sum(axis=1), "%s/losses/loss.png"%saveinto)

  AUCs = []

  for mass in [0.3, 0.4, 0.5, 0.8, 1.0]:
    train_df = data_handling.assignMassToBkg(train_df, mass)
    test_df = data_handling.assignMassToBkg(test_df, mass)
    X_train, y_train, X_test, y_test, w_train, w_test = data_handling.getXyw(train_df, test_df, [mass], train_features)

    if mass in model.train_masses:
      i = model.train_masses.index(mass)
      plotting.plotLoss(train_loss[:,i], test_loss[:,i], "%s/losses/loss_%d.png"%(saveinto,int(mass*1000)))
    plotting.plotOutputScore(model, X_train, y_train, w_train, "%s/outputScores/train_%d.png"%(saveinto,int(mass*1000)))
    plotting.plotOutputScore(model, X_test, y_test, w_test, "%s/outputScores/test_%d.png"%(saveinto,int(mass*1000)))
    train_score, test_score = plotting.plotAUC(model, X_train, y_train, X_test, y_test, w_train, w_test, "%s/ROC/ROC_%d.png"%(saveinto,int(mass*1000)))
    plotting.plotDistributions(model, X_train, y_train, w_train, "%s/distributions/%d/"%(saveinto,int(mass*1000)))

    #make dataframes for limit calculation
    sig_df = pd.concat([train_df, test_df])
    sig_df = sig_df[(sig_df.y==1)&(sig_df.mass==mass)]
    bkg_df = data_df.copy()
    bkg_df.loc[:,"mass"] = mass
    limit, optimal_boundary, max_boundary = getLimit(model, sig_df, data_df, train_features)

    AUCs.append([train_score, test_score, limit, optimal_boundary, max_boundary])

  return AUCs

def plotAllbutLosses(model, train_df, test_df, data_df, train_features, saveinto):
  AUCs = []

  for mass in [0.3, 0.4, 0.5, 0.8, 1.0]:
    train_df = data_handling.assignMassToBkg(train_df, mass)
    test_df = data_handling.assignMassToBkg(test_df, mass)
    X_train, y_train, X_test, y_test, w_train, w_test = data_handling.getXyw(train_df, test_df, [mass], train_features)

    if mass in model.train_masses:
      i = model.train_masses.index(mass)
    plotting.plotOutputScore(model, X_train, y_train, w_train, "%s/outputScores/train_%d.png"%(saveinto,int(mass*1000)))
    plotting.plotOutputScore(model, X_test, y_test, w_test, "%s/outputScores/test_%d.png"%(saveinto,int(mass*1000)))
    train_score, test_score = plotting.plotAUC(model, X_train, y_train, X_test, y_test, w_train, w_test, "%s/ROC/ROC_%d.png"%(saveinto,int(mass*1000)))
    plotting.plotDistributions(model, X_train, y_train, w_train, "%s/distributions/%d/"%(saveinto,int(mass*1000)))

    #make dataframes for limit calculation
    sig_df = pd.concat([train_df, test_df])
    sig_df = sig_df[(sig_df.y==1)&(sig_df.mass==mass)]
    bkg_df = data_df.copy()
    bkg_df.loc[:,"mass"] = mass
    limit, optimal_boundary, max_boundary = getLimit(model, sig_df, data_df, train_features)

    AUCs.append([train_score, test_score, limit, optimal_boundary, max_boundary])

  return AUCs

# def trainAndPlot(train_df, test_df, train_features, train_masses, saveinto):
#   model = paramNN.ParamNN(train_features, train_masses)
#   X_train, y_train, X_test, y_test, w_train, w_test = data_handling.getXyw(train_df, test_df, train_masses, train_features)
#   train_loss, test_loss = model.train(X_train, y_train, X_test, y_test, w_train, w_test, max_epochs=100, batch_size=128, lr=0.001, min_epoch=20, grace_epochs=10, tol=0.01, gamma=0.95)
#   AUCs = plotAll(model, train_df, test_df, train_features, train_loss, test_loss, saveinto)
#   return AUCs

def trainAndPlot(train_df, test_df, train_features, train_masses, saveinto):
  model = paramNN.BDT(train_features, train_masses)
  X_train, y_train, X_test, y_test, w_train, w_test = data_handling.getXyw(train_df, test_df, train_masses, train_features)
  model.train(X_train, y_train, X_test, y_test, w_train, w_test)
  AUCs = plotAllbutLosses(model, train_df, test_df, data_df, train_features, saveinto)
  return AUCs

def writeToAUCRecord(AUCs, filename, masses, title):
  with open(filename, "a") as f:
    strs = [title]
    for i in range(5):
      m = int(masses[i]*1000)
      strs.append("%d %.4f %.4f %.4f %.2f %.2f"%(m, AUCs[i][0], AUCs[i][1], AUCs[i][2], AUCs[i][3], AUCs[i][4]))
    strs.append("\n")
    f.write("\n".join(strs))

def quickTest(train_df, test_df, train_features):
  train_masses = [0.3, 0.4]
  trainAndPlot(train_df, test_df, train_features, train_masses, "quickTest/")

def performTests(train_df, test_df, train_features, name):
  masses = [0.3, 0.4, 0.5, 0.8, 1.0]
  test_dict = od()
  test_dict["all"] = masses
  for m in masses:
    masses_copy = masses.copy()
    masses_copy.remove(m)
    test_dict["all_except_%d"%int(m*1000)] = masses_copy
  for m in masses:
    test_dict["only_%d"%int(m*1000)] = [m]
  print(test_dict)

  for key in test_dict.keys():
    print(key)
    train_masses = test_dict[key]
    AUCs = trainAndPlot(train_df, test_df, train_features, train_masses, "plots/%s/%s/"%(name,key))
    writeToAUCRecord(AUCs, "plots/%s/AUC_record.txt"%name, masses, key)


train_df, test_df, data_df = data_handling.loadData(0.5)

features_init_test = ['tau1_pt', 'diphoton_delta_R']

features_1tau0lep_all =  [
  'nJet', 'MET_pt', 'MET_phi', 'diphoton_pt_mgg',
  'diphoton_rapidity', 'diphoton_delta_R', 'lead_pho_ptmgg',
  'sublead_pho_ptmgg', 'lead_pho_eta', 'sublead_pho_eta',
  'lead_pho_idmva', 'sublead_pho_idmva', 'lead_pho_phi',
  'sublead_pho_phi', 'tau1_pt', 'tau1_eta', 'tau1_phi',
  'tau1_id_vs_e', 'tau1_id_vs_m',
  'tau1_id_vs_j', 'jet1_pt', 'jet1_eta', 'jet1_id',
  'jet1_bTagDeepFlavB', 'jet2_pt', 'jet2_eta', 'jet2_id',
  'jet2_bTagDeepFlavB'
  ]

features_1tau0lep_nojet = [
  'MET_pt', 'MET_phi', 'diphoton_pt_mgg',
  'diphoton_rapidity', 'diphoton_delta_R', 'lead_pho_ptmgg',
  'sublead_pho_ptmgg', 'lead_pho_eta', 'sublead_pho_eta',
  'lead_pho_idmva', 'sublead_pho_idmva', 'lead_pho_phi',
  'sublead_pho_phi', 'tau1_pt', 'tau1_eta', 'tau1_phi',
  'tau1_id_vs_e', 'tau1_id_vs_m',
  'tau1_id_vs_j'
  ]

features_2tau_test = ['tau1_pt', 'diphoton_delta_R', 'reco_mX']

# train_features = features_init_test.copy()
# train_features.append("mass")
# quickTest(train_df, test_df, train_features)

train_features = features_1tau0lep_nojet.copy()
train_features.append("mass")
performTests(train_df, test_df, train_features, "1tau0lep_nojet_BDT_wLimits_gt100Bkg")

# train_features = features_2tau_test.copy()
# train_features.append("mass")
# performTests(train_df, test_df, train_features, "2tau_test")

# train_features = [
#   'nJet', 'MET_pt', 'MET_phi', 'diphoton_pt_mgg',
#   'diphoton_rapidity', 'diphoton_delta_R', 'lead_pho_ptmgg',
#   'sublead_pho_ptmgg', 'lead_pho_eta', 'sublead_pho_eta',
#   'lead_pho_idmva', 'sublead_pho_idmva', 'lead_pho_phi',
#   'sublead_pho_phi', 'tau1_pt', 'tau1_eta', 'tau1_phi',
#   'jet1_pt', 'jet1_eta', 'jet1_bTagDeepFlavB', 'jet2_pt', 'jet2_eta',
#   'jet2_bTagDeepFlavB',
#   'tau1_id_vs_e_3',
#   'tau1_id_vs_e_7', 'tau1_id_vs_e_15', 'tau1_id_vs_e_31',
#   'tau1_id_vs_e_63', 'tau1_id_vs_e_127', 'tau1_id_vs_e_255',
#   'tau1_id_vs_m_1', 'tau1_id_vs_m_3', 'tau1_id_vs_m_7', 'tau1_id_vs_m_15',
#   'tau1_id_vs_j_15', 'tau1_id_vs_j_31', 'tau1_id_vs_j_63',
#   'tau1_id_vs_j_127', 'tau1_id_vs_j_255', 'tau2_id_vs_e_-9',
#   'tau2_id_vs_m_-9', 'tau2_id_vs_j_-9', 'jet1_id_-9', 'jet1_id_0',
#   'jet1_id_2', 'jet1_id_6', 'jet2_id_-9', 'jet2_id_0', 'jet2_id_2',
#   'jet2_id_6'
#   ]
# train_features.append("mass")
# trainAndPlot(train_df, test_df, train_features, [0.3], "300/")

# import xgboost
# from sklearn import metrics

# xgbc = xgboost.XGBClassifier()

# train_df.loc[train_df.y==0,"weight"] *= 1/sum(train_df.loc[train_df.y==0,"weight"])
# train_df.loc[train_df.y==1,"weight"] *= 1/sum(train_df.loc[train_df.y==1,"weight"])

# #xgbc.fit(train_df[train_features], train_df.y, sample_weight=train_df.weight)
# xgbc.fit(train_df[train_features], train_df.y)

# x_test = test_df[train_features]
# y_test = test_df.y
# y_test_pred = xgbc.predict_proba(test_df[train_features])[:,1]

# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred, sample_weight=test_df.weight)
# #auc = metrics.auc(fpr, tpr)
# auc = np.trapz(tpr, fpr)
# print(auc)

"""
train_features = features_1tau0lep_nojet
train_features.append("mass")
train_masses = [0.3]

train_df = data_handling.assignMassToBkg(train_df, 0.3)
test_df = data_handling.assignMassToBkg(test_df, 0.3)
X_train, y_train, X_test, y_test, w_train, w_test = data_handling.getXyw(train_df, test_df, train_masses, train_features)

NN_model = paramNN.ParamNN(train_features, train_masses)
BDT_model = paramNN.BDT(train_features, train_masses)

#train_df.loc["weight"] = 1
#test_df.loc["weight"] = 1

train_loss, test_loss = NN_model.train(X_train, y_train, X_test, y_test, w_train, w_test, max_epochs=300, batch_size=128, lr=0.001, min_epoch=300, grace_epochs=10, tol=0.01, gamma=0.90)
#AUCs = plotAll(NN_model, train_df, test_df, train_features, train_loss, test_loss, "plots/NN_vs_BDT_test")

BDT_model.train(X_train, y_train, X_test, y_test, w_train, w_test)

print("NN", NN_model.getROC(X_train, y_train, w_train)[2])
print("BDT", BDT_model.getROC(X_train, y_train, w_train)[2])

print("NN", NN_model.getROC(X_test, y_test, w_test)[2])
print("BDT", BDT_model.getROC(X_test, y_test, w_test)[2])
"""