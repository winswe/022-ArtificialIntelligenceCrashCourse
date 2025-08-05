from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

file = 'data.csv'
data = np.asarray(pd.read_csv(file, header=None))
X = data[:,0:2]
Y = data[:,2]

#model = DecisionTreeClassfier(max_depth=5, min_samples_leaf=10)
model = DecisionTreeClassifier()
model.fit(X, Y)
y_pred = model.predict(X)
acc = accuracy_score(Y, y_pred)

print(acc)