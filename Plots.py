import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plot_x = []
resultsKNN = []
resultsRF = []
for j in range(1, 16):
    kbest = SelectKBest(f_classif, k=j)
    # Fitting SVM to the Training set
    from sklearn.svm import SVC

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    clf = Pipeline([
        ('feature_selection', kbest),
        ('classification', classifier)
    ])
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    result = scores.mean()

    plot_x.append(j)
    resultsKNN.append(result)

    classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
    clf = Pipeline([
        ('feature_selection', kbest),
        ('classification', classifier)
    ])
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    result = scores.mean()
    resultsRF.append(result)


ax = plt.subplot(2, 1, 1)
ax.set_title("Accuracy KNN - number of KBest ")
plt.plot(plot_x, resultsKNN)
ax = plt.subplot(2, 1, 2)
ax.set_title("Accuracy RFT - number of KBest", )
plt.plot(plot_x, resultsRF)
plt.show()