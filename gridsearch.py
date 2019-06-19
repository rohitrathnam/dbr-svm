from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import pandas as pd

print(__doc__)

# Loading the dataset
data = pd.read_csv('norm_dataset.csv', header=0)

# Split input output
X = data.loc[:,:'x18']
y = data['y']

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1)

# Construct grid
c=[0]*10
gamma=[0]*10
for i in range(0,10):
    c[i] = pow(2,-7+i)
for i in range(0,10):
    gamma[i] = pow(2,-5+i)

# Set the parameters by cross-validation
param_grid = [
#    {'kernel': ['linear'], 'C': c}
#    keeps increasing
#    {'kernel': ['rbf'], 'gamma': gamma, 'C': c}
#    c=8, gamma=1
#    {'kernel': ['sigmoid'], 'gamma': gamma, 'C': c}
#    c=2, gamma=2
#    {'kernel': ['poly'], 'gamma': gamma, 'C': c, 'degree': [2]}
#    c=0.125, gamma=64
#    {'kernel': ['poly'], 'gamma': gamma, 'C': c, 'degree': [3]}
#    c=0.015625, gamma=8
#    {'kernel': ['poly'], 'gamma': gamma, 'C': c, 'degree': [4]}
#    c=0.0078125, gamma=16
    ]

scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))
    print()
