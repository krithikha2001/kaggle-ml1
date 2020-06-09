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
import seaborn as sb
import matplotlib.pyplot as plt
sb.distplot(a=train[train['Survived']==1]['Pclass'],label='Alive')
sb.distplot(a=train[train['Survived']==0]['Pclass'],label='Dead')
plt.xticks([1,2,3])
#survival depends on pclass

#since age has null values
test['Age'].replace(np.nan,np.median(test['Age'].dropna()),inplace=True)
train['Age'].replace(np.nan,np.median(train['Age'].dropna()),inplace=True)
train.head()

train['Sex'].value_counts()
#most men die than woman
train['Embarked'].value_counts()
#depends on embarked place

sb.scatterplot(x=train['Survived'],y=train['SibSp'])
sb.scatterplot(x=train['Survived'],y=train['Parch'],color="red")
#both sibsp and patch do not deopend on survival]

coloum=["Pclass", "Sex", "SibSp", "Parch"]
X_test = pd.get_dummies(test[coloum])
X_train = pd.get_dummies(train[coloum])

y=train['Survived']  
basic = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
basic.fit(X_train, y)
#training model
y_pred = basic.predict(X_test)


answer = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
print(answer)

basic.score(X_train,y)
answer.to_csv('my_submission.csv', index=False)


