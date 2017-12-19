import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

data=pd.read_csv('data/titanic.csv')

columns_target= ['Survived']
columns_train = ['Age','Pclass','Sex','Fare']

X=data[columns_train]
Y=data[columns_target]

print X.head()
print X.info()


X['Sex'].isnull().sum()
X['Pclass'].isnull().sum()
X['Fare'].isnull().sum()
X['Age'].isnull().sum()


X['Age'] = X['Age'].fillna(X['Age'].median())

print X.describe()

X['Age'].isnull().sum()

d={'male':0, 'female':1}
X['Sex']=X['Sex'].apply(lambda x:d[x])

X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
clf=svm.LinearSVC()
clf.fit(X_train,Y_train)

print clf.predict(X_test[0:10])
print clf.score(X_test,Y_test)