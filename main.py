import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Handlers.LoadData import LoadData

data_set = LoadData().data_set

x = data_set[['Pclass', 'SibSp', 'Parch', 'Fare', 'processed_age', 'labeled_Sex', 'labeled_cabin']]
y = data_set['Survived']

X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=0)

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(max_iter=1000),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]

models = list(zip(names, classifiers))
predictions_df = pd.DataFrame()
predictions_df["Actual"] = y_test

# Try out Several algorithms independently looking for the most accurate model predictions
for name, model in models:
    print("==============={}=================".format(name))
    # Fit the dataset to the current algorithm
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    # append each prediction from the current algorithm to a dataframe for comparision
    predictions_df["{} Predictions".format(name)] = prediction

    # Generate the accuracy score for the current model
    score = model.score(X_test, y_test)
    print("{} Score: {}".format(name, score))

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

print("===============Voting Classifier=================")
votingClassifier = VotingClassifier(estimators=models, voting='hard', n_jobs=-1)
votingClassifier.fit(X_train, y_train)

print("Voting Classifier: Score: {}".format(votingClassifier.score(X_test, y_test)))
predictions_df["Voting Classifier Predictions"] = votingClassifier.predict(X_test)
prediction = votingClassifier.predict(X_test)

print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
