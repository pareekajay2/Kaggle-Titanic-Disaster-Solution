import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('train.csv')
X_train = data1.iloc[:, [1, 3, 4, 5, 6, 8]].values
y_train = data1.iloc[:, 11].values

data2 = pd.read_csv('test.csv')
X_test = data2.iloc[:, [1, 3, 4, 5, 6, 8]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'most_frequent')
imputer1 = imputer.fit(X_train[:, [2, 5]])
X_train[:, [2, 5]] = imputer1.transform(X_train[:, [2, 5]])
imputer2 = imputer.fit(X_test[:, [2, 5]])
X_test[:, [2, 5]] = imputer2.transform(X_test[:, [2, 5]])

from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X_train[:, 1] = LE1.fit_transform(X_train[:, 1])
LE2 = LabelEncoder()
X_test[:, 1] = LE2.fit_transform(X_test[:, 1])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)