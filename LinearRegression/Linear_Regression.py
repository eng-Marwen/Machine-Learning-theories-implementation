import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load CSV (use script's directory for relative path)
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "salary_data.csv"))

# Features and target
X = data[['YearsExperience']]   # must be 2D
y = data['Salary']
print(data)

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Parameters
print("Weight (w):", model.coef_[0])
print("Bias (b):", model.intercept_)

# Prediction
exp = 6
predicted_salary = model.predict([[exp]])
print("Predicted salary:", predicted_salary[0])
