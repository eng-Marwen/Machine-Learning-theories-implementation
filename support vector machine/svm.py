import pandas as pd
import numpy as np
import seaborn as sns
import os 
import matplotlib.pyplot as plt
from sklearn.svm  import SVC #support vector classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

# =================================================
# STEP 1: LOAD THE DATA
# =================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "diabetes.csv"))

# =================================================
# STEP 2: EXOLORE THE DATA
# =================================================

print(data.isnull().sum())

# =================================================
# STEP 3:PREPARE FEATURES AND TAARGET
# =================================================

y=data.Outcome
x=data.drop('Outcome',axis=1)
# print(x.head())
#normalize
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print(x_scaled)
# =================================================
# STEP 4: SPLIT THE DATA
# =================================================

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

# =================================================
# STEP 5: CREATE AND TRAIN THE SVM MODEL
# =================================================

#SVM classifier

svm=SVC(kernel='linear',C=1.0,gamma='scale',class_weight='balanced',random_state=42)
#C :error tolerence
#train
svm.fit(x_train,y_train)

# =================================================
# STEP 5: Evaluate the model
# =================================================

y_pred=svm.predict(x_test)

acc=accuracy_score(y_test,y_pred)
mat=classification_report(y_test,y_pred)

print(acc)
print(mat)