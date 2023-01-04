VERSION = 0.1

import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import xgboost as xgb
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict
import sys, getopt,os
from sklearn import metrics
import scipy as sp
from tqdm import tqdm
inputfile = ''
outputfile = ''

try:
	opts, args = getopt.getopt(sys.argv[1:],"hvi:o:")  
except getopt.GetoptError:
	print('Usage: python3 Reg.py\n  -i <input filename> (please use .csv files) \n  -o <output file>\n   [-h (Display this help message)]')
	sys.exit(2)                             
                                                                 
for opt, arg in opts:
	if opt == '-i':
		inputfile = arg
	elif opt == '-o':
		outputfile = arg
df = pd.read_csv(inputfile,dtype={'genome_id':str})
print('load ok')
X = df.iloc[0:,2:]
y = df['resistant_phenotype']
xg_reg = model = XGBClassifier(use_label_encoder=False,eval_metric="auc",n_estimators=500)
xg_reg.fit(X, y) 
feature_important = xg_reg.get_booster().get_score(importance_type='gain')
keys = list(feature_important.keys())
values = list(feature_important.values())
datagroup_v = pd.DataFrame(index=keys,data=values,columns=["score"]).sort_values(by = "score", ascending=False)

c=pd.DataFrame()
for iu in range(0,datagroup_v.shape[0],1):
	c = pd.concat([c, X[datagroup_v.index[iu]]], axis=1, names=datagroup_v.index[iu])

Best=0
point=0

aList = list()
for line in tqdm(range(0,c.shape[1],1)): 
	XX=c.iloc[:,:line+1]
	model = SVC(kernel='linear')
	scores = cross_val_score(model, XX, y, cv=10, scoring='roc_auc')
	#print("r2_score: %f" %r2_score(y,scores))
	aList.append(round(scores.mean(),4))
	if(round(scores.mean(),4)>Best):
		Best=round(scores.mean(),4)
		point=line
		print("Best mean: %f" %round(scores.mean(),4))
		print("Best point =",point+1)


c = c.iloc[0:,:point+1]
model = SVC(kernel='linear')
scores = cross_val_score(model, c, y, cv=10, scoring='roc_auc')
print("Extracted ", c.shape[1],"  features")
print("Classification use linear roc_auc of the dataset using extracted features is", round(scores.mean(),4));

outF = open(outputfile, "w")		
outF.write("Extracted ")
outF.write(str(c.shape[1]))
outF.write(" features\n")
outF.write("Classification roc_auc of the dataset using extracted features is ")
outF.write(str(round(scores.mean(),4)))
outF.write("\n")
outF.write("Extracted features:")
outF.write("\n")
for line in c.columns.values:
	outF.write(line)
	outF.write("\n")


outF.write("\n")


	

