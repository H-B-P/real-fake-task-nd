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

uselessCols=["id", #that's column order
"Gender", #modelling on this is illegal
"Customer", #discriminating against individual customers? no
"Effective To Date", #we predict the future, it will never be 2011 again, and since I'm guessing we cover the whole year seasonality is also pretty much useless
"Claim Reason", #we won't know claim reason until after claim is made, and we're trying to sell this
"State Code", #redundant with State
"Country"] #everywhere is America

df=df[[c for c in df.columns if c not in uselessCols]]

catCols  = ['State', 'Response', 'Coverage', 'Education', 'EmploymentStatus', 'Location Code', 'Marital Status', 'Policy Type', 'Policy', 'Sales Channel', 'Vehicle Class', 'Vehicle Size', "Number of Policies"]
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

models = [actual_modelling.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "Total Claim Amount",0.7), actual_modelling.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "Total Claim Amount", 0.2),actual_modelling.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "Total Claim Amount", 0.1)]
models = actual_modelling.construct_model(trainDf, "Total Claim Amount", 1, 0.05, penas, models)

#==Viz Model==
j=0

for model in models:
 for col in contCols:
  print(col)
  intervs, prevs = viz.get_cont_pdp_prevalences(trainDf, col)
  print (intervs)
  print (prevs)
  xs, ys = util.convert_lines_to_points(model, trainDf, col)
  viz.draw_cont_pdp(xs, ys, 0.2, "model_"+"ABCD"[j]+"_"+col+"_pdp")
  print(xs)
  print(ys)
 
 for col in catCols:
  print(col)
  print(viz.get_cat_pdp_prevalences(trainDf, col, 0.05))
  print(model["cats"][col])
 j+=1
#==Predict==

testDf["PREDICTED"]=pd.Series([0]*len(testDf))
for model in models:
 testDf["PREDICTED"]+=actual_modelling.predict(testDf, model)

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
