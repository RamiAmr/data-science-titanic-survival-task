# Importing the libraries

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sns.set()


def preprocess(df):
    # Drop un-useful columns
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df["Embarked"] = df["Embarked"].interpolate()
    df["Age"] = df["Age"].interpolate()
    df["Fare"] = df["Fare"].interpolate()
    # Select All columns for X except the Survived
    X = df.loc[:, df.columns != 'Survived']
    # X = X.loc[:, X.columns != 'PassengerId']

    # Encoding categorical data
    dummies = []
    cols = ['Pclass', 'Sex', 'Embarked']
    for col in cols:
        dummies.append(pd.get_dummies(X[col], drop_first=True))

    X = pd.concat((X,
                   pd.concat(dummies, axis=1)
                   ), axis=1)
    X = X.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

    # # Taking care of missing data
    # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # imputer.fit(X[["Age", "Fare"]])
    # X[["Age", "Fare"]] = imputer.transform(X[["Age", "Fare"]])

    # Feature Scaling
    # sc = StandardScaler()
    # X = sc.fit_transform(X)

    return X


# Importing the dataset

dataset = pd.read_csv("./data/train.csv")
dataset_test = pd.read_csv("./data/test.csv")

y = dataset["Survived"].values

X_train = preprocess(dataset)
X_train = X_train.loc[:, X_train.columns != 'PassengerId']
X_results = preprocess(dataset_test)

# Splitting the dataset into the Training set and Test set
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y, test_size=0.3, random_state=0)

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear", "GB"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=50, random_state=0),
    LogisticRegression(max_iter=1000, random_state=0),
    SGDClassifier(max_iter=100, random_state=0),
    MultinomialNB(),
    SVC(random_state=0),
    GradientBoostingClassifier(n_estimators=100, random_state=0)
]

models = list(zip(names, classifiers))
predictions_df = pd.DataFrame()
predictions_df["Actual"] = y_train_test

most_accurate_model = None
max_accuracy = 0
for name, model in models:
    print("==============={}=================".format(name))
    # Fit the dataset to the current algorithm
    model.fit(X_train_train, y_train_train)

    prediction = model.predict(X_train_test)
    # append each prediction from the current algorithm to a dataframe for comparision
    predictions_df["{} Predictions".format(name)] = prediction

    score = model.score(X_train_test, y_train_test)
    # Generate the accuracy score for the current model
    print("{} Score: {}".format(name, model.score(X_train_test, y_train_test)))

    if score > max_accuracy:
        max_accuracy = score
        most_accurate_model = model

    # print(classification_report(y_test, prediction))
    print(confusion_matrix(y_train_test, prediction))
    #     todo evaluate model based on CAP method instead of accuracy

print(most_accurate_model)
clf = most_accurate_model
y_results = clf.predict(X_results.iloc[:, 1:].values)
output = np.column_stack((X_results.iloc[:, 0].values, y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('./data/titanic_results.csv', index=False)
