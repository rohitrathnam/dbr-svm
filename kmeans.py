from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# To visualize the actual data in top 2 dimensions
iris=load_iris()
x,y=iris.data,iris.target

model=SVC().fit(x,y)
predicted=model.predict(x)

svd=TruncatedSVD().fit_transform(x)

from matplotlib import pyplot as plt
plt.figure(figsize=(16,6))
plt.subplot(1,2,0)
plt.title('Actual data, with errors highlighted')
colors=['r','g','b']
for t in [0,1,2]:
    plt.plot(svd[y==t][:,0],svd[y==t][:,1],colors[t]+'+')

errX,errY=svd[predicted!=y],y[predicted!=y]
for t in [0,1,2]:
    plt.plot(errX[errY==t][:,0],errX[errY==t][:,1],colors[t]+'o')


# To visualize the SVM classifier across
import numpy as np
density=15
domain=[np.linspace(min(x[:,i]),max(x[:,i]),num=density*4 if i==2 else density) for i in range(4)]

from itertools import product
allxs=list(product(*domain))
allys=model.predict(allxs)

allxs_svd=TruncatedSVD().fit_transform(allxs)

plt.subplot(1,2,1)
plt.title('Prediction space reduced to top two SVD\'s')
plt.ylim(-3,3)
for t in [0,1,2]:
    plt.scatter(allxs_svd[allys==t][:,0],allxs_svd[allys==t][:,1],color=colors[t],alpha=0.2/density,edgecolor='None')
