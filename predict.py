import pandas as pd
import numpy as np  
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv('student_performance_dataset.csv')

print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

plt.scatter(df['Study_Hours_per_Week'], df['Final_Exam_Score'])
plt.xlabel('Hours Studied per Week') 
plt.ylabel('Final_Exam_Score')
plt.title('Study_Hours_per_Week vs Final_Exam_Score')
plt.savefig('scatter_plot.png')
print("Scatter plot saved to scatter_plot.png")

print("Before removing:", len(df))

def remove_outliers_iqr(data,column):
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1

    lower=Q1- 1.5*IQR
    upper=Q3 + 1.5*IQR

    filtered=data[(data[column] >=lower) & (data[column] <= upper)]
    return filtered

df=remove_outliers_iqr(df,"Study_Hours_per_Week")
df=remove_outliers_iqr(df,"Final_Exam_Score")
print("After removing:", len(df))


plt.scatter(df['Study_Hours_per_Week'], df['Final_Exam_Score'])
plt.xlabel('Hours Studied per Week') 
plt.ylabel('Final_Exam_Score')
plt.title('Study_Hours_per_Week vs Final_Exam_Score')
plt.savefig('scatter_plot_after.png')
print("Scatter plot after removing Outliers saved to scatter_plot.png")


df_clean=df.copy()
df_clean['Extracurricular_Activities']=df_clean['Extracurricular_Activities'].map({'Yes':1,'No':0})

X = df_clean[['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores','Extracurricular_Activities']]
y = df_clean['Final_Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=3)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

model=LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Save the model
joblib.dump(model, 'student_performance_model.pkl')
