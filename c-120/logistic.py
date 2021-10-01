from os import stat_result
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")
X = df[["glucose", "bloodpressure"]]
Y = df["diabetes"]

x_train , x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=42)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)

y_prediction = logistic_regression.predict(x_test)
accuracy = accuracy_score(y_test,y_prediction)
print(accuracy)