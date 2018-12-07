import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

X = SelectKBest(chi2, k=2).fit_transform(X, y)


print(set(y))
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# To musi byc inaczej nie dziala
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n = 500
w0 = 20
w1 = 50
w2 = 50

X1 = np.array([random.uniform(-1, 1) for i in range(0, n)])
X2 = np.array([random.uniform(-1, 1) for j in range(0, n)])
y = [np.sign(w0+w1*X1[k]+w2*X2[k]) for k in range(0, n)]

classifiers = [GaussianNB(),
               LogisticRegression(random_state=0),
               KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
               DecisionTreeClassifier(criterion='entropy', random_state=0)]

titles = ["Bayes Gaussian", "LogisticRegression", "KNN", "Decision Tree"]
subplots = [221, 222, 223, 224]
plt.figure(1)
plt.subplots_adjust(hspace=0.4)
# X_train = np.array([X1.ravel(), X2.ravel()]).T
# y_train = y

from matplotlib import cm
for i, classifier in enumerate(classifiers):
    classifier.fit(X_train, y_train)
    X_set, y_set = X_train, y_train

    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))


    plt.subplot(subplots[i])
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.5, cmap=plt.cm.get_cmap('cubehelix', 26))

    plt.title(titles[i])

    for i, j in enumerate(np.unique(y_set)):
         plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c=plt.cm.get_cmap('cubehelix', 26)(i), label=j)
plt.show()


