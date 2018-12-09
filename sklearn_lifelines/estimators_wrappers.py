from lifelines import AalenAdditiveFitter
from lifelines import CoxPHFitter
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method

class CoxPHFitterModel(BaseEstimator):
    def __init__(self, duration_column=None, event_col=None, initial_beta=None, include_likelihood=False, strata=None, alpha=0.95, tie_method='Efron', normalize=True, penalizer=0.0, **kwargs):
        self.alpha = alpha
        self.tie_method = tie_method
        self.normalize = normalize
        self.penalizer = penalizer

        self.duration_column = duration_column
        self.event_col = event_col

        self.initial_beta = initial_beta
        self.include_likelihood = include_likelihood
        self.strata = strata


    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        X_[self.duration_column]=y[self.duration_column]
        if self.event_col is not None:
            X_[self.event_col] = y[self.event_col]

        params = self.get_params()
        est = CoxPHFitter(**params)

        est.fit(X_, duration_col=self.duration_column, event_col=self.event_col, initial_beta=self.initial_beta, include_likelihood=self.include_likelihood, strata=self.strata, **fit_params)
        self.estimator = est
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        return self.estimator.predict_expectation(X)[0].values[0]

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "tie_method": self.tie_method, "normalize": self.normalize, "penalizer": self.penalizer}


class AalenAdditiveFitterModel(BaseEstimator):

    def __init__(self, duration_column=None, event_col=None, timeline=None, id_col=None, fit_intercept=True, alpha=0.95, coef_penalizer=0.5, smoothing_penalizer=0.0,**kwargs):
        self.fit_intercept=fit_intercept
        self.alpha=alpha
        self.coef_penalizer=coef_penalizer
        self.smoothing_penalizer=smoothing_penalizer

        self.duration_column = duration_column
        self.event_col = event_col
        self.timeline = timeline
        self.id_col = id_col

    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        X_[self.duration_column]=y[self.duration_column]
        if self.event_col is not None:
            X_[self.event_col] = y[self.event_col]

        params = self.get_params()
        est = AalenAdditiveFitter(**params)
        est.fit(X_, duration_col=self.duration_column, event_col=self.event_col, timeline=self.timeline, id_col = self.id_col, **fit_params)
        self.estimator = est
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        return self.estimator.predict_expectation(X)[0].values[0]

    def get_params(self, deep=True):
        return {"fit_intercept": self.fit_intercept, "alpha": self.alpha, "coef_penalizer": self.coef_penalizer, "smoothing_penalizer": self.smoothing_penalizer}


