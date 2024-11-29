import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class SimpleAveragingClassifier(BaseEstimator, ClassifierMixin):

    CATEGORICAL_COL = "cell_type"

    def __init__(self, model, scale_data=False):
        """
        Simple Averaging Classifier: Combines the predictions of multiple models by averaging their predicted probabilities.

        Parameters:
        model: model
        """
        self.model = model
        self.scale_data = scale_data

    def scale_input(self, X):
        
        # Separate the column to exclude
        unscaled_data = X[SimpleAveragingClassifier.CATEGORICAL_COL]
        # Scale the remaining columns
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(X.drop(columns=[SimpleAveragingClassifier.CATEGORICAL_COL])),
            columns=X.drop(columns=[SimpleAveragingClassifier.CATEGORICAL_COL]).columns
        )
        # Recombine the scaled and unscaled columns
        X = pd.concat([scaled_data, unscaled_data.reset_index(drop=True)], axis=1)

        return X
    
    def fit(self, X, y):
        """
        Fit each model in the ensemble to the training data.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Training data
        y: array-like, shape (n_samples,)
            Target values
        """

        X_concat = pd.concat(X, ignore_index=True)    
        
        # scale data
        if self.scale_data:
            X_concat = self.scale_input(X_concat)


        y_concat = np.concat([[label]*len(df) for df,label in zip(X,y)])

        self.model.fit(X_concat, y_concat)

        self.classes_ = np.unique(y_concat)

        return self

    def predict_proba_wo_ensembling(self, X):
        """
        Predict class probabilities for each sample in X without ensembling results.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            Data to predict on

        Returns:
        prob: array-like, shape (n_samples, n_classes)
            Predicted class probabilities
        """

        X_concat = pd.concat(X, ignore_index=True)    
        
        # scale data
        if self.scale_data:
            X_concat = self.scale_input(X_concat)
        
        return self.model.predict_proba(X_concat)
    
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
        probs = np.zeros((len(X), 2))  # Initialize array to store probabilities
        for idx, elem in enumerate(X):
            # scale data
            if self.scale_data:
                input = self.scale_input(elem)
            else:
                input = elem

            probs[idx] = np.mean(self.model.predict_proba(input), axis=0)
        
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
