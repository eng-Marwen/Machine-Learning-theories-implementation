import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt 
import seaborn as sns


# Load CSV (use script's directory for relative path)
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "data.csv"))
#try to understand the date classification clean or no  ....
# print(data.info())
# print(data.describe())
# print(data.head())

#clean data
sns.heatmap(data.isnull())#Detect Missing Values 
# plt.show()

# Plot succeed vs study_hours
plt.figure(figsize=(8, 5))
plt.scatter(data['study_hours'], data['succeed'], c=data['succeed'], cmap='coolwarm', edgecolors='k')
plt.xlabel('Study Hours')
plt.ylabel('Succeed (0=Fail, 1=Success)')
plt.title('Study Hours vs Success')
plt.yticks([0, 1], ['Fail', 'Success'])
plt.show()

y=data['succeed']
x=data.drop('succeed',axis=1)
# print(y.head())
# print(x.head)

#normalize the data

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
x_scaled=scaler.fit_transform(x)

#split the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x_scaled,y ,test_size=0.3,random_state=42)

#training phase 
from sklearn.linear_model import Perceptron
 #create the model
perceptron=Perceptron()
#train it 
perceptron.fit(x_train,y_train)

#eval
y_pred=perceptron.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report
acc=accuracy_score(y_pred,y_test)

mat=classification_report(y_pred,y_test)
print(acc)
print(mat)


# Metric	Value	Interpretation
# Accuracy	92%	11 out of 12 predictions correct
# Precision	93% avg	When it predicts a class, it's usually right
# Recall	92% avg	It finds most of the actual cases
# F1-Score	92% avg	Good balance between precision and recall