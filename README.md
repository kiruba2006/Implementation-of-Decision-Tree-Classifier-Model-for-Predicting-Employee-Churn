# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load employee data and split it into training and testing sets.

2.Train a Decision Tree classifier using entropy as the split criterion.

3.Evaluate the model using accuracy, confusion matrix, and classification report.

4.Use the trained model to predict whether a new employee will stay or leave.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kiruba RC
RegisterNumber: 212224230125
*/
```

```python
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```


## Output:

<img width="1324" height="287" alt="Screenshot 2026-08-28 233007" src="https://github.com/user-attachments/assets/4c63c1c8-c098-41f0-b31e-6873d0d1e917" />

<img width="632" height="465" alt="Screenshot 2026-08-28 233834" src="https://github.com/user-attachments/assets/6d22563a-6987-4793-ac3e-d5892b9772b0" />

<img width="341" height="507" alt="Screenshot 2026-08-28 233650" src="https://github.com/user-attachments/assets/caa8e339-c652-4c7e-8f87-a801351104c2" />

<img width="1274" height="312" alt="Screenshot 2026-08-28 233909" src="https://github.com/user-attachments/assets/4faa63b4-f349-4fc1-8d59-5607df59c95e" />

<img width="1349" height="290" alt="Screenshot 2026-08-28 234001" src="https://github.com/user-attachments/assets/efb10989-da6b-41b2-accc-18e3cce5b7bf" />

```
Accuracy value:
0.9833333333333333
```

<img width="916" height="654" alt="Screenshot 2026-08-28 234113" src="https://github.com/user-attachments/assets/dadfb12d-a50e-4441-9c6e-9e2ffd260c31" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
