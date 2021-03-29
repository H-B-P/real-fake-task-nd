import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

def get_cont_pdp_prevalences(df, col, intervals=10, weightCol=None):
 cdf= df.copy()
 
 if type(intervals)==type([1,2,3]):
  intervs=intervals
 else:
  gap=(max(df[col])-min(df[col]))/float(intervals)
  print(min(df[col]), max(df[col]), gap)
  intervs=list(np.arange(min(df[col]), max(df[col])+gap, gap))
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 prevs=[]
 
 for i in range(len(intervs)-1):
  loInt = intervs[i]
  hiInt = intervs[i+1]
  if i==(len(intervs)-2):
   prevs.append(sum(cdf[(cdf[col]<=hiInt) & (cdf[col]>=loInt)]["weight"]))
  else:
   prevs.append(sum(cdf[(cdf[col]<hiInt) & (cdf[col]>=loInt)]["weight"]))
 
 return intervs, prevs

def get_cat_pdp_prevalences(df, col, threshold=0.05, weightCol=None):
 cdf= df.copy()
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 uniques = pd.unique(cdf[col])
 opDict = {}
 totalWeight = float(sum(cdf["weight"]))
 
 for unique in uniques:
  specWeight = float(sum(cdf[cdf[col]==unique]["weight"]))
  if (specWeight/totalWeight)>=threshold:
   opDict[unique] = specWeight
 
 opDict["OTHER"] = sum(cdf[~cdf[col].isin(opDict)]["weight"])
 
 return opDict

def get_Xiles(df, predCol, actCol, X=10):
 cdf = df.copy()
 cdf = cdf.sort_values([predCol, actCol])
 cdf = cdf.reset_index(drop=True)
 preds = []
 acts = []
 for i in range(X):
  lowerlim = int(((i)*len(cdf))/X)
  upperlim = int(((i+1)*len(cdf))/X)
  subset = cdf[lowerlim:upperlim]
  preds.append(sum(subset[predCol])/len(subset))
  acts.append(sum(subset[actCol])/len(subset))
 return preds, acts

def draw_cont_pdp(X, Y, targetSpan=0, name="graph"):
 layout = {
  "yaxis": {
    "range": [min(min(Y), 1-targetSpan), max(max(Y), 1+targetSpan)]
  }
 }
 
 fig = go.Figure(
    data=go.Data([X,Y]), layout=layout
)
 
 plotly.offline.plot(fig, filename=name+'.html')
