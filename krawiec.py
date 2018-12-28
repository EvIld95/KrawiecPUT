# Importing the libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

poly = PolynomialFeatures(2)

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

kbest = SelectKBest(f_classif, k=8)

X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy SVM:', accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation KNN: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy KNN:', accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation DecisionTree: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy DecisionTree:', accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation RandomForestTree: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy RandomForestTrees:', accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation Bayes: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy Bayes:', accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
clf = Pipeline([
    ('feature_selection', kbest),
    ('classification', classifier)
])
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy CrossValidation LogisticRegression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Accuracy LogisticRegression:', accuracy_score(y_test, y_pred))
