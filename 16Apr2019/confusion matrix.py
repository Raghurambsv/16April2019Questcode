from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import numpy as np
from statistics import mode
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

######K=1
print('###K=1')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred))#96.67% accuracy


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))
# print the classification_report
print(metrics.classification_report(y_test, y_pred))

######K=11
print('###K=11')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred))#96.67% accuracy


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

######K=13
print('###K=13')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred))#96.67% accuracy


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# print the confusion matrix
print(metrics.confusion_matrix(y_test, y_pred))

#print(metrics.classification_report(y_test, y_pred))



