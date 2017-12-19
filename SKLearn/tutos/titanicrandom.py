from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

X = pd.read_csv('data/titanic.csv')
y = X.pop('Survived')

X['Age'].fillna(X.Age.mean(), inplace=True) # tots els forats els canvies per el promig

numeric_variables = list(X.dtypes[X.dtypes != "object"].index) #agafa nomes les comunes que tenen valors

model = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=42)
model.fit(X[numeric_variables], y)

model.oob_score_

y_oob = model.oob_prediction_
print "c-stat: ", roc_auc_score(y,y_oob)

X.drop(['Name','Ticket','PassengerId'], axis=1,inplace=True)

def clean_cabin(x):
	try:
		return x[0]
	except TypeError:
		return 'None'

X['Cabin'] = X.Cabin.apply(clean_cabin)

categorical_variables = ['Sex','Cabin','Embarked']

for variable in categorical_variables:
	X[variable].fillna('Missing', inplace=True)
	dummies = pd.get_dummies(X[variable], prefix=variable)
	X = pd.concat([X, dummies], axis=1)
	X.drop([variable], axis=1,inplace=True)

model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features='auto',min_samples_leaf=5)
model.fit(X,y)

y_oob = model.oob_prediction_
print "c-stat: ", roc_auc_score(y,y_oob)

model.feature_importances_

#ensenya la importancia de cada parametre
#feature_importances = pd.Series(model.feature_importances_,index=X.columns)
#feature_importances.sort()
#feature_importances.plot(kind='barh', figsize=(7,6))
