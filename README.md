# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Nithya shree B
RegisterNumber:  212233220071  
*/
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```
## Output:
![decision tree classifier model](sam.png)

![Screenshot 2024-04-15 090828](https://github.com/Balunithu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161273477/77052b1c-0587-41a4-99d8-910b42ded6b1)

![Screenshot 2024-04-15 090943](https://github.com/Balunithu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161273477/f782a37b-34a0-4fa6-9b07-ab3d2bdfeb5e)

![Screenshot 2024-04-15 091219](https://github.com/Balunithu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161273477/05bfdff1-1a2f-47a8-9a38-977b01859e5a)

![Screenshot 2024-04-15 091327](https://github.com/Balunithu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161273477/e754fe20-d524-473e-8e84-f660132d8ae1)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
