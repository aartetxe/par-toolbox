from sklearn.base import BaseEstimator
import numpy as np


class ResamplingClassifier(BaseEstimator):
    """
    Resampling classifier

    The resampler must implement imblearn's 'fit_sample' method

    Example
    -------

    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.under_sampling import RandomUnderSampler

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    rus = RandomUnderSampler()
    dt = DecisionTreeClassifier()

    rus_dt = ResamplingClassifier(base_classifier=dt, resampler=rus)

    rus_dt.fit(X,y)
    print rus_dt.predict(X)

    """
    def __init__(self,
                 base_classifier=None,
                 resampler=None):

        self.base_classifier = base_classifier
        self.resampler = resampler
        
    def fit(self, X, y):

        classes_ = np.unique(y)
        self.classes_ = classes_
        self.n_classes_ = len(classes_)

        n_samples, self.n_features_ = X.shape

        # Preprocess original data
        resampler = self.resampler
        X_res, y_res = resampler.fit_sample(X, y)
        self.X_res = X_res
        self.y_res = y_res
            
        return self
        
    def predict(self, X):
        
        X_res = self.X_res
        y_res = self.y_res
        base_classifier = self.base_classifier
            
        base_classifier.fit(X_res, y_res)
            
        return base_classifier.predict(X)

    def predict_proba(self, X):

        X_res = self.X_res
        y_res = self.y_res
        base_classifier = self.base_classifier

        base_classifier.fit(X_res, y_res)

        return base_classifier.predict_proba(X)