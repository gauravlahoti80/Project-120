import pandas as pd
from os import stat_result
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("income.csv")
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]] 
Y = df["income"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25, random_state=42)
standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)
guassian_nb = GaussianNB()
guassian_nb.fit(x_train,y_train)

y_prediction = guassian_nb.predict(x_test)
accuracy = accuracy_score(y_test,y_prediction)
print(accuracy)