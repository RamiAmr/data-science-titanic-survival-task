import pandas as pd
from pprint import pprint as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.TitanicRF import TitanRandomForest
from models.TitanicLogic import TitanicLogic
from Handlers.LoadData import LoadData


data_set = LoadData().data_set

x = data_set[['Pclass', 'SibSp', 'Parch', 'Fare', 'processed_age', 'labeled_Sex', 'labeled_cabin']]
y = data_set['Survived']

X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=0)

TitanicRF = TitanRandomForest(X_train, y_train, X_test, y_test)

TitanLog = TitanicLogic(X_train, y_train, X_test, y_test)
pp('----------- score ----------')
pp(TitanicRF.get_score())

pp('----------- score Logisitc ----------')
pp(TitanLog.get_score())
