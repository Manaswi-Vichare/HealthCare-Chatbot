import numpy as np
import pandas as pd
from sklearn import linear_model
import joblib
data = pd.read_csv("data/diabetes.csv")
print(data.head())
logreg = linear_model.LogisticRegression()
X=data.iloc[:,:8]
print(X.shape[1])
y=data[["Outcome"]]
X=np.array(X)
y=np.array(y)
logreg.fit(X,y.reshape(-1,))
joblib.dump(logreg,"model1")
