from sklearn.cross_validation import train_test_split
import pandas as pd
from modeltf import model

dataset = pd.read_csv("train.csv", sep = ",", header = 0)
testset = pd.read_csv("test.csv", sep = ",", header = 0)
X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, 0]
m = X.shape[0]

X, X_cv, Y, Y_cv = train_test_split(X.values, Y.values, test_size = 0.25)

_, _, parameters = model(X, Y, X_cv, Y_cv)