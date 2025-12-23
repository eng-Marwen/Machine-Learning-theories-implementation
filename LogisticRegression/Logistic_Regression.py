import pandas as pd
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV (use script's directory for relative path)
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "data.csv"))


#information about the dataset
# print(data.tail())
# print(data.info())
# print(data.describe())
#data cleaining 
sns.heatmap(data.isnull())
#
#we see that unamed variable has null values so we will drop it with the ids
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
#replace categorical values with numerical values
data.diagnosis = [1 if value == 'M' else 0 for value in data.diagnosis]
#devide the data into features and target
y = data['diagnosis']
X = data.drop('diagnosis', axis=1)

#normalization : some values might be very large or very small, so we scale the features for the model
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


#train the model 
from sklearn.linear_model import LogisticRegression

#create an isnstance of the model

lr=LogisticRegression()

lr.fit(X_train,y_train)

#predict the target variable based on the test data(tetsing our model)

y_pred=lr.predict(X_test)


#evaluation of the model

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import classification_report
cl=classification_report(y_pred,y_test)
print(cl)
