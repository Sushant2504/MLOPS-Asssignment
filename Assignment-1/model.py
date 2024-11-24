import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HousePricePredictor:
    def __init__(self, data_path):
        self.data_path = "../data/sample_submission.csv"
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def preprocess_data(self):
        X = self.data[['feature1', 'feature2']]  
        y = self.data['price']  

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict_price(self, features):
        
        features = np.array(features).reshape(1, -1)

        scaled_features = self.scaler.transform(features)

        predicted_price = self.model.predict(scaled_features)[0]
        return predicted_price