# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

print(tf.__version__)

sns.set()


def init_ann(X, y):
    # init the ANN
    ann = tf.keras.models.Sequential()
    # create the first hidden layer with 6 nodes
    ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
    # create another hidden layer with 6 nodes
    ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
    # create the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    # Compile the ANN
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Train the ANN
    ann.fit(X, y, batch_size=32, epochs=200)

    return ann


def preprocess(df):
    global cabin_imputer, numeric_imputer, embarked_imputer, sex_le, ct, sc

    num_vars = ["Age", "Fare"]
    numeric_imputer.fit(df[num_vars])
    df[num_vars] = numeric_imputer.transform(df[num_vars])
    cabin_imputer.fit(df[["Cabin"]])
    df[["Cabin"]] = cabin_imputer.transform(df[["Cabin"]])
    # Remove all cabin numbers and leave out only the cabin letters indicating what level on the titanic was that person
    df["Cabin"] = df["Cabin"].apply(lambda x: ''.join(i.lower() for i in x if not i.isdigit()))
    # handel several cabin letters that where generated ex. B58 F60 -> b f -> b
    df["Cabin"] = df["Cabin"].apply(lambda x: x.strip()[0])
    embarked_imputer.fit(df[["Embarked"]])
    df[["Embarked"]] = embarked_imputer.transform(df[["Embarked"]])
    df["Embarked"] = df["Embarked"].str.lower()
    # encode Gender Column using label encoder
    sex_le.fit(df["Sex"])
    df["Sex"] = sex_le.transform(df["Sex"])
    # encode Embarked column using OneHotEncoder
    ct.fit(df)
    df = ct.transform(df)
    # Feature Scaling (All inputs to an ANN Must Be Scaled down)
    sc.fit(df)
    df = sc.transform(df)

    return df


def preprocess_test(df):
    global cabin_imputer, numeric_imputer, embarked_imputer, sex_le, ct, sc

    num_vars = ["Age", "Fare"]

    df[num_vars] = numeric_imputer.transform(df[num_vars])

    df[["Cabin"]] = cabin_imputer.transform(df[["Cabin"]])
    # Remove all cabin numbers and leave out only the cabin letters indicating what level on the titanic was that person
    df["Cabin"] = df["Cabin"].apply(lambda x: ''.join(i.lower() for i in x if not i.isdigit()))
    # handel several cabin letters that where generated ex. B58 F60 -> b f -> b
    df["Cabin"] = df["Cabin"].apply(lambda x: x.strip()[0])

    df[["Embarked"]] = embarked_imputer.transform(df[["Embarked"]])
    df["Embarked"] = df["Embarked"].str.lower()
    # encode Gender Column using label encoder
    df["Sex"] = sex_le.transform(df["Sex"])
    # encode Embarked column using OneHotEncoder
    df = ct.transform(df)
    # Feature Scaling (All inputs to an ANN Must Be Scaled down)
    df = sc.transform(df)

    return df


# Importing the dataset
dataset = pd.read_csv("./data/train.csv")

y = dataset["Survived"].values

# Drop un-useful columns
X = dataset.drop(['PassengerId', "Survived", 'Name', 'Ticket'], axis=1)

# fill missing values
# TODO figure out the optimal way to fill the missing values

embarked_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
cabin_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="na")
numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
ct = ColumnTransformer(transformers=[
    # ('ss', StandardScaler(), ['max_temp',
    #                           'avg_temp',
    #                           'min_temp']),
    ('encoder', OneHotEncoder(), ["Embarked", "Cabin"])
], remainder='passthrough')
sex_le = LabelEncoder()
sc = StandardScaler()

X = preprocess(X)

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = init_ann(X, y)

# y_pred = model.predict(X)
# y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
#
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print(accuracy_score(y_test, y_pred))

dataset_test = pd.read_csv("./data/test.csv")

X_results = preprocess_test(
    dataset_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1).copy()
)

y_results = model.predict(X_results)
y_results = (y_results > 0.5)
# y_results = model.predict(X_results.iloc[:, 1:].values)
output = np.column_stack((dataset_test.iloc[:, 0].values, y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('./data/titanic_results.csv', index=False)

