import pandas as pd
import numpy as np
import sklearn as sk

raw = pd.read_csv('dataset.csv')
data = raw.copy()

for i in range(0,17):
	maxval = max(raw.iloc[:,i])
	minval = min(raw.iloc[:,i])
	for j in range(0,1150):
		data.iloc[j,i] = (raw.iloc[j,i] - minval) / (maxval - minval)
data.to_csv('norm_dataset.csv')
