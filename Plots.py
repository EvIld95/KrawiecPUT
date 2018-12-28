import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib

from SelectByTrainingNBest import SelectByTrainingNBest
from SelectMedianNBest import SelectMedianNBest

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
plot_x = []

###################

resultsSTB = []
for j in range(3, 12):
    stb = SelectByTrainingNBest(n_best=j, X=X_train, y=y_train)
    train = stb.transform()

    # Fitting SVM to the Training set
    from sklearn.svm import SVC

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    scores = cross_val_score(classifier, train, y_train, cv=2)
    result = scores.mean()
    resultsSTB.append(result)

###########################

resultsSKB = []
for j in range(3, 12):
    smb = SelectMedianNBest(n_best=j, X=X_train, y=y_train)
    train = smb.transform()

    # Fitting SVM to the Training set
    from sklearn.svm import SVC

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    scores = cross_val_score(classifier, train, y_train, cv=3)
    result = scores.mean()
    resultsSKB.append(result)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

resultsKNN = []
resultsRF = []
for j in range(3, 12):
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

    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    clf = Pipeline([
        ('feature_selection', kbest),
        ('classification', classifier)
    ])
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    result = scores.mean()
    resultsRF.append(result)


# ax = plt.subplot(3, 1, 1)
# ax.set_title("Accuracy KNN - number of KBest ")
# plt.plot(plot_x, resultsKNN)
# ax = plt.subplot(3, 1, 2)
# ax.set_title("Accuracy RFT - number of KBest", )
# plt.plot(plot_x, resultsRF)
# ax = plt.subplot(3, 1, 3)
# ax.set_title("Accuracy SKB", )
# plt.plot(plot_x, resultsSKB)

plt.plot(plot_x, resultsKNN, label='KNN')
plt.plot(plot_x, resultsRF, label='RF')
plt.plot(plot_x, resultsSKB, label='SKB')
plt.plot(plot_x, resultsSTB, label='STB')
plt.legend(loc='upper left')
plt.show()



