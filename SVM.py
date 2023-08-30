from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.clf = SVC(kernel=self.kernel)
    
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.clf.predict(X_test)