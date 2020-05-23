from sklearn.linear_model import LogisticRegression

class TitanicLogic:
    x_train: object
    y_train: object

    X_test: object
    y_test: object
    score: float
    data: list
    model: object
    fit: object
    predict: object

    def __init__(self, x_train, y_train, X_test, y_test):
        self.x_trian = x_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.modiling()

    def modiling(self):
        self.model = LogisticRegression()
        self.fit = self.model.fit(self.x_trian, self.y_train)

    def make_prediction(self, X_test):
        self.predict = self.model.predict(X_test)
        return self.predict

    def get_score(self ):
        return self.model.score(self.X_test, self.y_test)