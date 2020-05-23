import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class LoadData:
    data_set: pd.DataFrame

    def __init__(self):
        self.data_set = pd.read_csv('./data/train.csv')
        self.data_set = self.proccess_columns()


    def proccess_columns(self):
        self.data_set['processed_age'] = self.data_set['Age'].fillna(self.data_set['Age'].median())
        self.data_set['processed_cabin'] = self.data_set['Cabin'].fillna('sundeck')
        self.data_set['labeled_Sex'] = self.data_set['Sex'].apply(lambda x: 2 if x == 'male' else 1)
        enco = LabelEncoder()
        self.data_set['labeled_cabin'] = enco.fit_transform(self.data_set['processed_cabin'])
        self.data_set = self.drop_columns()
        return self.data_set

    def drop_columns(self):
        to_be_dropped = ['Name', 'Ticket', 'Embarked', 'Cabin', 'Age', 'Sex', 'processed_cabin']
        self.data_set = self.data_set.drop(to_be_dropped, axis=1)
        return self.data_set

