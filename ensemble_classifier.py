import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleAveragingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        """
        Simple Averaging Classifier: Combines the predictions of multiple models by averaging their predicted probabilities.

        Parameters:
        model: model
        """
        self.model = model
    
    def fit(self, X, y):
        """
        Fit each model in the ensemble to the training data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Training data
        y: array-like, shape (n_samples,)
            Target values
        """

        self.model.fit(X, y)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for each sample in X by averaging the probabilities from each model.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Data to predict on

        Returns:
        prob: array-like, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        # Get the predicted probabilities from each model
        probs = np.zeros((len(X), 2))  # Initialize array to store probabilities
        for idx, elem in enumerate(X):
            probs[idx] = np.mean(self.model.predict_proba(elem), axis=0)
        
        # Average the probabilities
        return np.array(probs)
    
    def predict(self, X):
        """
        Predict the class labels for the input samples by using the averaged probabilities.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Data to predict on

        Returns:
        y_pred: array-like, shape (n_samples,)
            Predicted class labels
        """
        # Get the averaged probabilities
        prob = self.predict_proba(X)

        # Return the class with the highest averaged probability
        return np.argmax(prob, axis=1)


def main():
    # Example usage:

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Example data
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize base models
    model = LogisticRegression(max_iter=500)

    # Create and train the Simple Averaging Classifier
    ensemble_model = SimpleAveragingClassifier(model=model)
    ensemble_model.fit(X_train_scaled, y_train)

    # Get predictions
    y_pred = ensemble_model.predict(X_test_scaled)

    # Print predictions
    print(y_pred)
