from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreparation:
    def __init__(self, data, target, train_size, test_size, val_size):
        self.target = target
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        #self.scaler =  MinMaxScaler()
        self.scaler =  StandardScaler()
    
    def split_data(self):
        train_end = int(self.data.shape[0] * self.train_size)
        test_end = train_end + int(self.data.shape[0] * self.test_size)

        X_train, y_train = self.data[:train_end], self.target[:train_end]
        X_test, y_test = self.data[train_end:test_end], self.target[train_end:test_end]
        X_val, y_val = self.data[test_end:], self.target[test_end:]

        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def prepare(self):
        X_train, X_test, X_val, y_train, y_test, y_val = self.split_data()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)

        return X_train, X_test, X_val, y_train, y_test, y_val


