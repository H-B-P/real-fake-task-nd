import pandas as pd
import numpy as np
import math
import actual_modelling

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def get_gradient(x, y):
 aveX = sum(x)/len(x)
 aveY = sum(y)/len(y)
 
 numerator=sum((x-aveX)*(y-aveY))
 denominator=sum((x-aveX)*(x-aveX))
 
 return numerator/denominator

def convert_lines_to_points(model, df, col):
 xlo=min(df[col])
 xhi=max(df[col])
 #get the xs
 xs = [xlo]
 for seg in model["conts"][col]["segs"]:
  xs.append(seg[0])
 xs.append(xhi)
 xs.sort()
 #get the ys
 ys = []
 for x in xs:
  ys.append(actual_modelling.get_effect_of_this_cont_col(df, model, col, x))
 return xs, ys

def convert_points_to_lines(xs, ys, z, col):
 op={"c":0, "z":z, "m":0}
 if len(xs)>2:
  segs=[]
  for i in range(1,len(xs)-2):
   pregrad = (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
   postgrad = (ys[i+1]-ys[i])/(xs[i+1]-xs[i])
   segs.append([xs[i],(postgrad-pregrad)/2])
  op["segs"] = segs
  p0 = 0
  p1 = 0
  for seg in segs:
   p0 += abs(xs[0]-seg[0])*seg[1]
   p1 += abs(xs[1]-seg[0])*seg[1]
 else:
  p0 = 0
  p1 = 0
 op["m"] = (ys[1]-p1)-(ys[0]-p0)/(xs[1]-xs[0])
 op["c"] = ys[0]-p0-op["m"]*(xs[0]-z)
 return op
