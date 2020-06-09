# kaggle-ml1

First we will read test.cv and train.cv files through pandas library.
Then I will check which all features are contributing to survival of passengers.

### Plotting graphs for pclass we get those graphs where upper class ppl have more survival rate
Through Seaborn plots we plot for Pclass and SibSp and Parch
By counting total we get that males died more than females and embarked place matters too  for survival
### age has null values so we fill that with median values and drop null values
## Using Random Forest Classifies modelbased on construction of several "trees" that will individually consider each passenger's data and vote on whether the individual survived.
Whichever tree gets most votes win!

So, we will consider features of Pclass ,Sex, SibSp, Parch,Age,Embarked
and using Eanodm Forest Classifier we train the model for x_train value and y=survived coloumn.
Then the trained model will predit y for x_test values.
This way we can get the desired output with passenger_id and y(predicted).


