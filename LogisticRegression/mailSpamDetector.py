import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression


# Load CSV (use script's directory for relative path)
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "mail_data.csv"))

# print(data.describe())
# print(sns.heatmap(data.isnull()))
# plt.show()
# print(data.head())

data.Category=[1 if value =='ham' else 0 for value in data.Category]
# print (data.Category)

#transform text data to numerical numbers
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)

x=vectorizer.fit_transform(data['Message'])
y=data.Category
print(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# What random_state does:
# Aspect	Explanation
# Purpose	Sets a seed for the random number generator
# With value (e.g., 42)	Same split every time you run the code
# Without it (None)	Different random split each run


#model training

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)

#evaluation

y_pred=lr.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report

acc=accuracy_score(y_pred,y_test)
CR=classification_report(y_pred,y_test)

print(acc)
print(CR)