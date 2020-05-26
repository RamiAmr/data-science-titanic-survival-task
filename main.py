# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print(tf.__version__)

sns.set()


def init_ann(X, y, hidden_layers=2, epochs=200):
    # init the ANN
    ann = tf.keras.models.Sequential()

    for i in range(hidden_layers):
        ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

    # create the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    # Compile the ANN
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Train the ANN
    ann.fit(X, y, batch_size=32, epochs=epochs)

    return ann


def preprocess(df):
    # Drop un-useful columns
    df = df.drop(['PassengerId', "Survived", 'Name', 'Ticket', "Cabin"], axis=1)

    df = preprocessor.fit_transform(df)
    df = sc.fit_transform(df)
    return df


def preprocess_test(df):
    # Drop un-useful columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', "Cabin"], axis=1)

    df = preprocessor.transform(df)
    df = sc.transform(df)
    return df


def cabin_column_handler(df):
    cabin_col_ndx = 3
    for i in range(len(df[:, cabin_col_ndx])):
        df[:, cabin_col_ndx][i] = ''.join(i.lower() for i in df[:, cabin_col_ndx][i] if not i.isdigit())
        df[:, cabin_col_ndx][i] = df[:, cabin_col_ndx][i].strip()[0]
    return df


numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ['Embarked', 'Sex', 'Pclass']

sc = StandardScaler()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    # ('cabin_handler', FunctionTransformer(cabin_column_handler)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(missing_values=np.nan)),
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features),
    ],
    remainder='passthrough'
)

# Importing the dataset
dataset = pd.read_csv("./data/train.csv")

y = dataset["Survived"].values

X = preprocess(dataset.copy())

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = init_ann(X_train, y_train, hidden_layers=2, epochs=200)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

dataset_test = pd.read_csv("./data/test.csv")

X_results = preprocess_test(dataset_test.copy())

y_results = model.predict(X_results)
y_results = (y_results > 0.5)

output = np.column_stack((dataset_test.iloc[:, 0].values, y_results))
df_results = pd.DataFrame(output.astype('int'), columns=['PassengerID', 'Survived'])
df_results.to_csv('./data/titanic_results.csv', index=False)
