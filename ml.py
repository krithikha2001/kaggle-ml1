# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
cols = ['Name','Ticket','Cabin']
train = train.drop(cols,axis=1)
test=test.drop(cols,axis=1)
'''dummies = []
train['Age'] = train['Age'].interpolate()
cols = ['Pclass','Sex','Embarked','Age']
for col in cols:
 dummies.append(pd.get_dummies(train[col]))
titanic_dummies = pd.concat(dummies, axis=1)
train = pd.concat((train,titanic_dummies),axis=1)
test = pd.concat((test,titanic_dummies),axis=1)

train = train.drop(['Pclass','Sex','Embarked','Age'],axis=1)
test = test.drop(['Pclass','Sex','Embarked','Age'],axis=1)
'''

#getting values only from these values
coloum=["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train[coloum])
X_test = pd.get_dummies(test[coloum])
#from sklearn.model_selection import train_test_split

#X_train=train[["Pclass", "Sex", "SibSp", "Parch"]]  # Features
y=train['Survived']  # Labels

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#creating gaussian classifier
basic = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
basic.fit(X_train, y)
#training model
y_pred = basic.predict(X_test)


answer = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
print(answer)

basic.score(X_train,y)
#answer.to_csv('my_submission.csv', index=False)
