import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

os.system("cls")

df = pd.read_csv("income.csv")
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]] 
Y = df["income"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25, random_state=0)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)

y_prediction = logistic_regression.predict(x_test)
accuracy = accuracy_score(y_test,y_prediction)
print(accuracy)