import numpy as np
import pandas as pd

import prep
import seg
import actual_modelling
import analytics
import viz
import util

SEGS_PER_CONT=5

#==Load in==

df = pd.read_csv("../data/new_train.csv")

uselessCols=["id", "Gender", "Customer", "Effective To Date", "Claim Reason", "State Code"]

df=df[[c for c in df.columns if c not in uselessCols]]

catCols  = ['Country', 'State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Sales Channel', 'Vehicle Class', 'Vehicle Size', "Number of Policies"]
contCols = [c for c in df.columns if c not in catCols+["Total Claim Amount"]]


#temporary code to test a trick

if False:
 for c in contCols:
  df[c]= df[c]/df[c].std(ddof=0)
  df[c]= df[c]-df[c].mean()


#==Random Split==

trainDf = df.sample(frac = 0.8, random_state=1) 
testDf = df.drop(trainDf.index)

trainDf = trainDf.reset_index()
testDf = testDf.reset_index()

#==Prep==

print("PREPARING")

#Categoricals

uniques={}

if True:
 for c in catCols:
  print(c)
  uniques[c] = prep.get_uniques_for_this_cat_col(trainDf,c, 0.05)

#Segmentation

segPoints={}

for col in contCols:
 print(col)
 
 ratioList = seg.get_ratios(SEGS_PER_CONT)
 segPointList = []
 for ratio in ratioList:
  segpt = seg.get_segpt(trainDf, col, ratio)
  roundedSegpt = util.round_to_sf(segpt, 3)
  if roundedSegpt not in segPointList and (roundedSegpt>min(df[col])) and (roundedSegpt<max(df[col])):
   segPointList.append(roundedSegpt)
 segPointList.sort()
 segPoints[col]=segPointList

#==Actually model!===

penas = {"segs":0.0, "grads":0.0, "contfeat":0.0, "catfeat":0.0,"uniques":0.0}

trainDfRural = trainDf[trainDf["Location Code"]=="Rural"]
trainDfUrban = trainDf[trainDf["Location Code"]=="Urban"]
trainDfSuburban = trainDf[trainDf["Location Code"]=="Suburban"]
#trainDfNotRural = trainDf.drop(trainDfRural.index)

trainDfRural = trainDfRural.reset_index()
trainDfUrban = trainDfUrban.reset_index()
trainDfSuburban = trainDfSuburban.reset_index()
#trainDfNotRural = trainDfNotRural.reset_index()

modelsRural = [actual_modelling.prep_starting_model(trainDfRural, contCols, segPoints, catCols, uniques, "Total Claim Amount")]
modelsRural = actual_modelling.construct_model(trainDfRural, "Total Claim Amount", 500, 0.05, penas, modelsRural)

modelsUrban = [actual_modelling.prep_starting_model(trainDfUrban, contCols, segPoints, catCols, uniques, "Total Claim Amount")]
modelsUrban = actual_modelling.construct_model(trainDfUrban, "Total Claim Amount", 500, 0.05, penas, modelsUrban)

modelsSuburban = [actual_modelling.prep_starting_model(trainDfSuburban, contCols, segPoints, catCols, uniques, "Total Claim Amount")]
modelsSuburban = actual_modelling.construct_model(trainDfSuburban, "Total Claim Amount", 500, 0.05, penas, modelsSuburban)

#modelsNotRural = [actual_modelling.prep_starting_model(trainDfNotRural, contCols, segPoints, catCols, uniques, "Total Claim Amount")]
#modelsNotRural = actual_modelling.construct_model(trainDfNotRural, "Total Claim Amount", 200, 0.05, penas, modelsNotRural)

#==Viz Model==

models = modelsRural+modelsUrban+modelsSuburban

for model in models:
 print("BIG_C:", model["BIG_C"])
 for col in contCols:
  print(col)
  #intervs, prevs = viz.get_cont_pdp_prevalences(trainDf, col) #TODO FIX THIS SO IT HANDLES ZEROS NICE LIKE
  xs, ys = util.convert_lines_to_points(model, trainDf, col)
  print (xs)
  print(ys)
 
 for col in catCols:
  print(col)
  print(model["cats"][col])

#==Predict==


testDfRural = testDf[testDf["Location Code"]=="Rural"]
testDfUrban = testDf[testDf["Location Code"]=="Urban"]
testDfSuburban = testDf[testDf["Location Code"]=="Suburban"]

testDfRural = testDfRural.reset_index()
testDfUrban = testDfUrban.reset_index()
testDfSuburban = testDfSuburban.reset_index()


testDfRural["PREDICTED"]=pd.Series([0]*len(testDfRural))
for model in modelsRural:
 testDfRural["PREDICTED"]+=actual_modelling.predict(testDfRural, model)

testDfUrban["PREDICTED"]=pd.Series([0]*len(testDfUrban))
for model in modelsUrban:
 testDfUrban["PREDICTED"]+=actual_modelling.predict(testDfUrban, model)

testDfSuburban["PREDICTED"]=pd.Series([0]*len(testDfSuburban))
for model in modelsSuburban:
 testDfSuburban["PREDICTED"]+=actual_modelling.predict(testDfSuburban, model)

#testDfNotRural["PREDICTED"]=pd.Series([0]*len(testDfNotRural))
#for model in modelsNotRural:
# testDfNotRural["PREDICTED"]+=actual_modelling.predict(testDfNotRural, model)

testDf = testDfRural.append(testDfUrban)
testDf = testDf.append(testDfSuburban)

testDf = testDf.reset_index(drop=True)

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
