import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# X = SelectKBest(f_classif, k=2).fit_transform(X, y)
X = X[:, [0,1]]

#print(X.shape, y.shape)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# To musi byc inaczej nie dziala
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifiers = [GaussianNB(),
               LogisticRegression(random_state=0),
               KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
               DecisionTreeClassifier(criterion='entropy', random_state=0),
               RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)]

titles = ["Bayes Gaussian", "LogisticRegression", "KNN", "Decision Tree", "RandomForestTree"]
subplots = [321, 322, 323, 324, 325]
plt.figure(1)
plt.subplots_adjust(hspace=0.4)

from matplotlib import cm
for i, classifier in enumerate(classifiers):
    classifier.fit(X_train, y_train)
    X_set, y_set = X_train, y_train

    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))


    plt.subplot(subplots[i])
    predictions = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    print(set(predictions))
    plt.contourf(X1, X2, predictions.reshape(X1.shape), alpha=0.5)

    plt.title(titles[i])

    #for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c=plt.cm.get_cmap('cubehelix', 26)(i), label=j)
plt.show()


