from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self,data, target, train_size, test_size, val_size):
        self.X = data
        self.y = target
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = StandardScaler()

    def split_data(self):
        train_end = int(self.X.shape[0] * self.train_size)
        test_end = train_end + int(self.X.shape[0] * self.test_size)
        X_train, y_train = self.X[:train_end], self.y[:train_end]
        X_test, y_test = self.X[train_end:test_end], self.y[train_end:test_end]
        X_val, y_val = self.X[test_end:], self.y[test_end:]
        return X_train, y_train, X_test, y_test, X_val, y_val

    def preprocess(self):
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_data()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)
        return X_train, y_train, X_test, y_test, X_val, y_val