import numpy as np
import pandas as pd
import xgboost as xgb


import prep
import seg
import actual_modelling
import analytics
import viz
import util

#==Load in==

df = pd.read_csv("../data/new_train.csv")

uselessCols=["id", "Gender", "Customer", "Effective To Date", "Claim Reason"]

df=df[[c for c in df.columns if c not in uselessCols]]

catCols  = ['Country', 'State Code', 'State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Sales Channel', 'Vehicle Class', 'Vehicle Size', "Number of Policies"]
contCols = [c for c in df.columns if c not in catCols+["Total Claim Amount"]]


#==Random Split==

trainDf = df.sample(frac = 0.8, random_state=1) 
testDf = df.drop(trainDf.index)

trainDf = trainDf.reset_index()
testDf = testDf.reset_index()

#==Prep==

print("PREPARING")

#Categoricals

for c in catCols:
 print(c)
 trainDf, uniques = prep.dummy_this_cat_col(trainDf,c, 0.05)
 testDf = prep.dummy_this_cat_col_given_uniques(testDf, c, uniques)

#==Actually model!===

explanatoryCols = list(trainDf.columns)
explanatoryCols.remove("Total Claim Amount")
dtrain = xgb.DMatrix(trainDf[explanatoryCols], label=trainDf["Total Claim Amount"])

params={'max_depth':3, "objective":"reg:gamma"}
bst = xgb.train(params, dtrain, 100)
 


#==Viz Model==

#let's not

#==Predict==

dtest = xgb.DMatrix(testDf[explanatoryCols], label=testDf["Total Claim Amount"])
preds = np.array(bst.predict(dtest))

testDf["PREDICTED"]=bst.predict(dtest)

print(testDf[["Total Claim Amount","PREDICTED"]])

#==Viz Predictions==

p, a = viz.get_Xiles(testDf, "PREDICTED", "Total Claim Amount", 10)
print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])

#==Analyze (i.e. get summary stats)==

print("MAE")
print(util.round_to_sf(analytics.get_mae(testDf["PREDICTED"],testDf["Total Claim Amount"])))
print("RMSE")
print(util.round_to_sf(analytics.get_rmse(testDf["PREDICTED"],testDf["Total Claim Amount"])))
print("MEANS")
print(util.round_to_sf(testDf["PREDICTED"].mean()), util.round_to_sf(testDf["Total Claim Amount"].mean()))
print("DRIFT COEFF")
#print(analytics.get_drift_coeff(testDf["PREDICTED"],testDf["loss"]))
print(util.round_to_sf(analytics.get_drift_coeff_macro(p,a)))
