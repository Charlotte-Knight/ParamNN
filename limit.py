import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import bisect
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.stats import expon
from scipy.stats import norm
from scipy.integrate import quad

#np.random.seed(20)

from tqdm import tqdm

import warnings

from functools import singledispatch

"""---Limit Setting------------------------------------------------------------------------------------------------"""

def calculateExpectedCLs(mu, s, b):
  """
  Calculates CLs for a given mu and expected number of signal and background events, s and b.
  Will calculate for a single category where s and b are input as just numbers.
  Will also calculate for multiple categories where s and b are arrays (entry per category)
  """
  s, b = np.array(s), np.array(b)
  qmu = -2 * np.sum( b*(np.log(mu*s+b) - np.log(b)) - mu*s )
  CLs = 1 - chi2.cdf(qmu, 1)
  return CLs

def calculateExpectedLimit(s, b, rlow=0, rhigh=10, plot=False):
  if calculateExpectedCLs(rhigh, s, b) < 0.05:
    return bisect(lambda x: calculateExpectedCLs(x, s, b)-0.05, rlow, rhigh, rtol=0.001)
  else:
    warnings.warn("Limit above rhigh = %f"%rhigh)
    return rhigh

  if plot:
    mus = np.linspace(rlow, rhigh, n)
    CLs = [calculateExpectedCLs(mu, s, b) for mu in mus]
    plt.plot(mus, CLs)
    plt.plot([limit,limit], [0, 0.05], 'k')
    plt.plot([0, limit], [0.05, 0.05], 'k')
    plt.show()

  return limit

"""----------------------------------------------------------------------------------------------------------------"""
"""---Fitting------------------------------------------------------------------------------------------------------"""

class ExpFunc():
  def __init__(self, N, norm, l, l_up=0, l_down=0):
    self.l = l
    self.l_up = l_up
    self.l_down = l_down

    self.N = N
    self.N_up = N + np.sqrt(N)
    self.N_down = N - np.sqrt(N)

    self.norm = norm
  
  def __call__(self, m, fluctuation=None):
    return (self.N / self.norm(self.l)) * np.exp(-self.l*m)

  def getLowerUpper(self, m):
    fluctuations = []
    for i, l in enumerate([self.l, self.l_up, self.l_down]):
      for j, N in enumerate([self.N, self.N_up, self.N_down]):
        fluctuations.append((N / self.norm(l)) * np.exp(-l*m))
    fluctuations = np.array(fluctuations)
     
    return np.min(fluctuations, axis=0), np.max(fluctuations, axis=0)

  def getNEventsInSR(self, sr):
    nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))

    fluctuations = []
    for i, l in enumerate([self.l, self.l_up, self.l_down]):
      for j, N in enumerate([self.N, self.N_up, self.N_down]):
        fluctuations.append(intExp(sr[0], sr[1], l, N/self.norm(l)))

    return nominal, min(fluctuations), max(fluctuations)

def intExp(a, b, l, N=1):
  return (N/l) * (np.exp(-l*a) - np.exp(-l*b))

def bkgNLL(l, bkg, norm=lambda l: 1/l):
  """
  Negative log likelihood from an unbinned fit of the background to an exponential.
  The exponential probability distribution: P(m, l) = [1/N(l)] * exp(-l*m)
  where m is the diphoton mass, l is a free parameter and N(l) normalises the distribution.

  bkg: a DataFrame of background events with two columns: "mass" and "weight"
  norm: a function that normalises P. This has to be specific to the range of masses
        considered, e.g. the sidebands but not the signal.

  WARNING: weighting not implemented yet
  """
  return np.mean(l*bkg.mass-np.log(1/norm(l)))

def fitBkg(bkg, pres, sr, l_guess):
  #fit only in sidebands
  m = bkg.mass
  bkg = bkg[((m>pres[0])&(m<sr[0])) | ((m>sr[1])&(m<pres[1]))]

  norm = lambda l: (intExp(pres[0], sr[0], l) + intExp(sr[1], pres[1], l))
  res = minimize(bkgNLL, l_guess, args=(bkg, norm))
  assert res.success, "Nbkg = %f \n"%len(bkg) + str(res)
  l_fit = res.x[0]
  #l_err = np.sqrt(res.hess_inv/len(bkg))

  ls = np.linspace(0, 0.1, 1000)
  
  f = lambda x: len(bkg)*(bkgNLL(x, bkg, norm)-bkgNLL(l_fit, bkg, norm)) - 0.5
  l_up = root(f, l_fit + np.sqrt(res.hess_inv/len(bkg))).x[0]
  l_down = root(f, l_fit - np.sqrt(res.hess_inv/len(bkg))).x[0]

  #bkg_func(m) = No. bkg events / GeV at a given value of m
  #N = lambda l: float(sum(bkg.weight))/norm(l)
  N = float(sum(bkg.weight))
  bkg_func = ExpFunc(N, norm, l_fit, l_up, l_down)

  #Number of bkg events in signal region found from fit
  nbkg_sr, nbkg_sr_down, nbkg_sr_up = bkg_func.getNEventsInSR(sr)
  #print(nbkg_sr-nbkg_sr_down, nbkg_sr_up-nbkg_sr)
  #print(nbkg_sr, nbkg_sr_down, nbkg_sr_up)

  return bkg_func, nbkg_sr, [nbkg_sr_down, nbkg_sr_up]

def performFit(sig, bkg, pres=(100,150), sr=(120,130), l_guess=0.1):
  """
  Return the number of signal and background events in the signal region.
  Number of signal events found by simply summing the number found in signal region.
  Number of background events found from an exponential fit to the side bands.

  sig, bkg: DataFrames of signal and background events with two columns: "mass" and "weight"
  pres: The preselection window on the diphoton mass
  sr: The signal region
  l_guess: An initial guess for the l parameter in the exponential fit
  """

  #background fit
  bkg_func, nbkg_sr, nbkg_ci = fitBkg(bkg, pres, sr, l_guess)

  #just sum signal events in signal region
  nsig_sr = sum(sig[ (sig.mass>sr[0])&(sig.mass<sr[1]) ].weight)

  return nsig_sr, nbkg_sr, nbkg_ci, bkg_func

def plotBkgFit(bkg, bkg_func, pres, sr, saveas="bkg_fit.png"):
  bkg = bkg[((bkg.mass>pres[0])&(bkg.mass<sr[0])) | ((bkg.mass>sr[1])&(bkg.mass<pres[1]))]
    
  m = np.linspace(pres[0], pres[1], 100)
  n, bin_edges = np.histogram(bkg.mass, bins=pres[1]-pres[0], range=(pres[0], pres[1]))
  bin_centers = np.array( (bin_edges[:-1] + bin_edges[1:]) / 2 )
  err = np.sqrt(n)
  err[err==0] = 1

  side_bands = (bin_centers<sr[0])|(bin_centers>sr[1])
  plt.errorbar(bin_centers[side_bands], n[side_bands], err[side_bands], fmt='o')
  plt.plot(m, bkg_func(m), label=r"$N = %.1f \cdot $exp$(-(%.3f^{+%.3f}_{-%.3f})*m_{\gamma\gamma})$"%(bkg_func.N, bkg_func.l, bkg_func.l_up-bkg_func.l, bkg_func.l-bkg_func.l_down))
  lower, upper = bkg_func.getLowerUpper(m)
  plt.fill_between(m, lower, upper, color="yellow", alpha=0.5)
  plt.title("Background fit")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("No. events")
  plt.ylim(bottom=0)
  plt.legend()
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

"""---Category Optimisation----------------------------------------------------------------------------------------"""

def AMS(s, b):
  """
  Calculate AMS for expected number of signal and background events, s and b.
  Will calculate for a single category where s and b are input as just numbers.
  Will also calculate for multiple categories where s and b are arrays (entry per category)
  """
  s, b = np.array(s), np.array(b)
  AMS2 = 2 * ( (s+b)*np.log(1+s/b) -s ) #calculate squared AMS for each category
  AMS = np.sqrt(np.sum(AMS2)) #square root the sum (summing in quadrature)
  return AMS


def optimiseBoundary(bkg, sig, pres=(100,150), sr=(120,130), low=0.05, high=1.0, step=0.01):
  boundaries = np.linspace(low, high, int((high-low)/step)+1)
  limits = []
  ams = []
  for i, bound in tqdm(enumerate(boundaries), leave=False):
    #split events according to boundary
    bkg_pass = bkg[bkg.score>bound]
    sig_pass = sig[sig.score>bound]
    bkg_npass = bkg[bkg.score<=bound]
    sig_npass = sig[sig.score<=bound]

    #reject boundary if too few events in high purity category
    side_bands = ((bkg_pass.mass>pres[0])&(bkg_pass.mass<sr[0])) | ((bkg_pass.mass>sr[1])&(bkg_pass.mass<pres[1]))
    if (len(bkg_pass[side_bands]) < 100) | (len(sig_pass) < 100):
      boundaries = boundaries[:i]
      print(pass_fit_nbkg)
      break
    
    pass_fit_nsig, pass_fit_nbkg, pass_fit_nbkg_ci, pass_bkg_func = performFit(sig_pass, bkg_pass, pres, sr)
    npass_fit_nsig, npass_fit_nbkg, npass_fit_nbkg_ci, npass_bkg_func = performFit(sig_npass, bkg_npass, pres, sr)
    
    #print(bound, fit_nsig, fit_nbkg)
    limits.append(calculateExpectedLimit([pass_fit_nsig, npass_fit_nsig], [pass_fit_nbkg, npass_fit_nbkg]))
    #limits.append(calculateExpectedLimit([pass_fit_nsig, npass_fit_nsig], [pass_fit_nbkg_ci[1], npass_fit_nbkg_ci[1]]))
    ams.append(AMS([pass_fit_nsig, npass_fit_nsig], [pass_fit_nbkg, npass_fit_nbkg]))

  limits = np.array(limits)
  optimal_boundary = boundaries[limits.argmin()]
  optimal_limit = limits.min()
  ams = np.array(ams)

  print(optimal_boundary, boundaries[-1])

  return optimal_limit, optimal_boundary, boundaries, limits, ams

"""----------------------------------------------------------------------------------------------------------------"""
"""---Testing------------------------------------------------------------------------------------------------------"""

def generateToyScores(nbkg=100, nsig=100, lbkg=1, lsig=10):
  nbkg_to_sample = int(1.5 * nbkg * 1/intExp(0, 1, lbkg))
  nsig_to_sample = int(1.5 * nsig * 1/intExp(0, 1, lsig))

  bkg_scores = expon.rvs(size=nbkg_to_sample, scale=1/lbkg)
  sig_scores = -1*expon.rvs(size=nsig_to_sample, scale=1/lsig) + 1

  bkg_scores = bkg_scores[(bkg_scores>0)&(bkg_scores<1)]
  sig_scores = sig_scores[(sig_scores>0)&(sig_scores<1)]

  return bkg_scores[:nbkg], sig_scores[:nsig]

def generateToyData(nbkg=100, nsig=100, l=0.05, mh=125, sig=1, pres=(100,150)):
  #need to sample a greater number of background events since we cut away with preselection
  int_pres = (np.exp(-l*pres[0]) - np.exp(-l*pres[1])) #integral of exponential in preselection window
  nbkg_to_sample = int(1.5 * nbkg * 1/int_pres)
  nsig_to_sample = int(1.5 * nsig)

  #sample mass distributions
  bkg_mass = expon.rvs(size=nbkg_to_sample, scale=1/l)
  sig_mass = norm.rvs(size=nsig_to_sample, loc=mh, scale=sig)

  #apply preselection
  bkg_mass = bkg_mass[(bkg_mass>pres[0])&(bkg_mass<pres[1])][:nbkg]
  sig_mass = sig_mass[(sig_mass>pres[0])&(sig_mass<pres[1])][:nsig]

  bkg_scores, sig_scores = generateToyScores(nbkg, nsig)
  
  #make DataFrames with events given unity weight
  bkg = pd.DataFrame({"mass":bkg_mass, "weight":np.ones(nbkg), "score":bkg_scores})
  sig = pd.DataFrame({"mass":sig_mass, "weight":np.ones(nsig), "score":sig_scores})

  return bkg, sig

def plotSigPlusBkg(bkg, sig, pres, saveas="bkg_sig.png"):
  plt.hist([bkg.mass, sig.mass], bins=50, range=pres, stacked=True, histtype='step', label=["background", "signal"])
  plt.title("Toy experiment")
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel("No. events")
  plt.legend()
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

def plotScores(bkg, sig, optimal_boundaries=None, labels=None, saveas="output_scores.png"):
  plt.hist(bkg.score, bins=50, range=(0,1), weights=bkg.weight, histtype='step', label="bkg", density=True)
  plt.hist(sig.score, bins=50, range=(0,1), weights=sig.weight, histtype='step', label="sig", density=True)

  if optimal_boundaries != None:
    for i, bound in enumerate(optimal_boundaries):
      plt.plot([bound, bound], [0, plt.ylim()[1]], '--', label=labels[i])

  plt.xlabel("Output score")
  plt.ylabel("No. Events (normalised)")
  plt.yscale("log")
  #plt.ylim(top=plt.ylim()[1]*10)
  plt.legend(loc='upper left')
  plt.savefig(saveas)
  #plt.show()
  plt.clf()

def testFit(nbkg=100, nsig=100, l=0.05, pres=(100,150), sr=(120,130)):
  bkg, sig = generateToyData(nbkg, nsig, l=l, pres=pres)
  plotSigPlusBkg(bkg, sig, pres, saveas="test_fit_sig_bkg.png")

  true_nsig = sum(sig[(sig.mass>sr[0])&(sig.mass<sr[1])].weight)
  true_nbkg = sum(bkg[(bkg.mass>sr[0])&(bkg.mass<sr[1])].weight)

  fit_nsig, fit_nbkg, fit_nbkg_ci, bkg_func = performFit(sig, bkg, pres, sr)
  print("True (fit) nsig: %d (%f)"%(true_nsig, fit_nsig))
  print("True (fit) nbkg: %d (%f)"%(true_nbkg, fit_nbkg))

  plotBkgFit(bkg, bkg_func, pres, sr, saveas="test_fit_bkg_fit.png")

  limit = calculateExpectedLimit(fit_nsig, fit_nbkg)
  print("95%% CL limit on mu: %f"%limit)

  if (true_nbkg>(fit_nbkg_ci[0]-np.sqrt(fit_nbkg_ci[0]))) & (true_nbkg<(fit_nbkg_ci[1]+np.sqrt(fit_nbkg_ci[1]))):
    return True

def testOptimisation(nbkg=100, nsig=100, l=0.05, pres=(100,150), sr=(120,130)):
  bkg, sig = generateToyData(nbkg, nsig, l=l, pres=pres)
  plotSigPlusBkg(bkg, sig, pres, saveas="test_optimisation_sig_bkg.png")

  plotScores(bkg, sig, saveas="test_optimisation_scores_no_boundaries.png")
  optimal_limit, optimal_boundary, boundaries, limits, ams = optimiseBoundary(bkg, sig, low=0.5, high=1.0)
  plotScores(bkg, sig, optimal_boundaries=[boundaries[limits.argmin()], boundaries[ams.argmax()]], labels=["CLs optimal boundary", "AMS optimal boundary"], saveas="test_optimisation_scores_w_boundaries.png")
  
  line = plt.plot(boundaries, (limits-min(limits))/max(limits), label="CLs")
  plt.plot([boundaries[limits.argmin()],boundaries[limits.argmin()]], [0, 1.1], '--', color=line[0]._color)

  for sf in [0.001, 0.01, 0.1, 1, 10,100]:
    sig_scaled = sig.copy()
    sig_scaled.loc[:,"weight"] *= sf
    optimal_limit, optimal_boundary, boundaries, limits, ams = optimiseBoundary(bkg, sig_scaled, low=0.5, high=0.95, step=0.005)
    line = plt.plot(boundaries, (ams-max(ams))/max(-ams), label="AMS sf=%.2f"%sf)
    plt.plot([boundaries[ams.argmax()],boundaries[ams.argmax()]], [0, 1], '--', color=line[0]._color)

    #line = plt.plot(boundaries, (limits-min(limits))/max(limits), label="CLs sf=%.2f"%sf)
    #line = plt.plot(boundaries, limits*sf, label="95%% CL Limit sf=%f"%sf)
    #plt.plot([boundaries[limits.argmin()],boundaries[limits.argmin()]], [0, plt.ylim()[1]], '--', color=line[0]._color)
  
  plt.legend()
  plt.xlabel("Output Score")
  plt.ylabel("Normalised performance metric")
  plt.savefig("test_optimisation_scores_norm_check.png")
  #plt.show()
  plt.clf()

def basicGGTTLimit():
  import xgboost as xgb
  import pickle
  from sklearn.model_selection import train_test_split

  path = "Pass1/"
  files = ["HHggtautau_ResonantPresel_1tau0lep_2018.pkl", "HHggtautau_ResonantPresel_1tau1lep_2018.pkl", "HHggtautau_ResonantPresel_2tau_2018.pkl"]
    
  dfs = []
  for filename in files:
    print("> %s"%filename)
    with open(path+filename, "rb") as f:
      dfs.append(pickle.load(f))
  df = pd.concat(dfs, ignore_index=True)
  
  df["y"] = 0
  df.loc[df.process_id<0, "y"] = 1

  #select mX = 500
  MC = df[(df.process_id==-5)|(df.process_id>0)]
  data = df[df.process_id==0]

  train_features = list(MC.columns)
  train_features.remove("gg_mass")
  train_features.remove("process_id")
  train_features.remove("y")
  train_features.remove("weight")
  print(train_features)

  train_df, test_df = train_test_split(MC, test_size=0.5, random_state=1)

  train_df.loc[train_df.y==1, "weight"] *= train_df.loc[train_df.y==0, "weight"].sum() / train_df.loc[train_df.y==1, "weight"].sum()
  model = xgb.XGBClassifier()
  model.fit(train_df[train_features], train_df["y"], sample_weight=train_df["weight"])

  sig_scores = model.predict_proba(MC[MC.y==1][train_features])[:,1]
  bkg_scores = model.predict_proba(data[train_features])[:,1]

  sig = pd.DataFrame({"mass": MC[MC.y==1].gg_mass, "score": sig_scores, "weight": MC[MC.y==1].weight})
  bkg = pd.DataFrame({"mass": data.gg_mass, "score": bkg_scores, "weight": data.weight})

  optimal_limit, optimal_boundary, boundaries, limits, ams = optimiseBoundary(bkg, sig, low=0.9, high=1.0, step=0.001)
  print(optimal_boundary)

  pres, sr = (100,150), (120,130)
  fit_nsig, fit_nbkg, fit_nbkg_ci, bkg_func = performFit(sig[sig.score>optimal_boundary], bkg[bkg.score>optimal_boundary], pres, sr)
  plotBkgFit(bkg[bkg.score>optimal_boundary], bkg_func, pres, sr, saveas="ggtt_high_purity_bkg_fit.png")
  fit_nsig, fit_nbkg, fit_nbkg_ci, bkg_func = performFit(sig[sig.score<=optimal_boundary], bkg[bkg.score<=optimal_boundary], pres, sr)
  plotBkgFit(bkg[bkg.score<=optimal_boundary], bkg_func, pres, sr, saveas="ggtt_low_purity_bkg_fit.png")

  print("Limit: %f"%optimal_limit)

  return MC, train_df

if __name__=="__main__":
  testFit(30, 30, l=0.01)
  #testOptimisation(10000, 10000)
  #bkg, sig = basicGGTTLimit()
