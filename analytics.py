import pandas as pd
import numpy as np
import math

def get_rmse(pred, act):
 err = pred-act
 return math.sqrt(sum(err*err)/len(err))

def get_mae(pred,act):
 err = pred-act
 return sum(abs(err))/len(err)

def get_means(pred,act):
 return sum(pred)/len(pred), sum(act)/len(act)

def get_drift_coeff_macro(predXiles, actXiles):
 avePred = sum(predXiles)/len(predXiles)
 aveAct = sum(actXiles)/len(actXiles)
 
 numerator=0
 denominator=0
 
 for i in range(len(predXiles)):
  numerator+=(predXiles[i]-avePred)*(actXiles[i]-aveAct)
  denominator+=(predXiles[i]-avePred)*(predXiles[i]-avePred)
 
 return numerator/denominator
