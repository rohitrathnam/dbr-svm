import pandas as pd
from scipy import interp
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle
import numpy as np
from sklearn.model_selection import train_test_split
from bokeh.plotting import figure
from bokeh.io import show

data = pd.read_csv('norm_dataset.csv')

X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state=0)

classifier = svm.SVC(kernel='rbf', gamma=1.0, C=8.0, probability=True)
classifier.fit(X_train, y_train)
y_score = classifier.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=2)

p = figure(plot_width=400, plot_height=400)
p.line(fpr, tpr, line_width=2)
show(p)
