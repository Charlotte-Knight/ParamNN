import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier as GBC
import matplotlib.pyplot as plt
import os

def plotHists(train_df, features):
    for feature in features:
        plt.hist(train_df[train_df.y==0][feature], bins=100, range=(-3,3), label="bkg", alpha=0.5)
        plt.hist(train_df[train_df.y==1][feature], bins=100, range=(-3,3), label="bsig", alpha=0.5)
        plt.xlabel(feature)
        plt.legend()
        plt.savefig("plots/highdimExample/%s.png"%feature)
        plt.clf()

print("Loading train data")
with open("/vols/cms/mdk16/ggtt/ParamNN/data/all_train.pkl", "rb") as f:
    train_df = pickle.load(f)
print("Loading test data")
with open("/vols/cms/mdk16/ggtt/ParamNN/data/all_test.pkl", "rb") as f:
    test_df = pickle.load(f)

try:
  os.makedirs("plots/highdimExample")
except:
  pass

plt.hist(train_df[train_df.y==0]["f26"], bins=100, range=(-3,3), label="bkg", alpha=0.5)
for mass in train_df.mass.unique():
    plt.hist(train_df[(train_df.y==1)&(train_df.mass==mass)]["f26"], bins=100, range=(-3,3), label=str(mass), alpha=0.5)
plt.legend()
plt.xlabel("mX")
plt.savefig("plots/highdimExample/mX.png")
plt.clf()

#select mass
train_df = train_df[train_df.mass==1000]
test_df = test_df[test_df.mass==1000]

features = list(filter(lambda x: "f" in x, train_df.columns))

plotHists(train_df, features)

X_train = train_df[features]
y_train = train_df["y"]

X_test = test_df[features]
y_test = test_df["y"]

clf = GBC(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0)
print("Training")
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
print(clf.feature_importances_)
