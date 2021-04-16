# ---------------------------------------------
# Calibration of predicted probabilities.
# ---------------------------------------------
import numpy as np
import pandas as pd
import sklearn

from . import utils

class SoftMaxCalibration:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        self._softmax = utils.get_softmax(zs, ys)

    def calibrate(self, zs):
        return self._softmax(zs)

class SigmoidCalibration:
    def __init__(self, num_calibration, num_bins):
        self._num_calibration = num_calibration
        self._num_bins = num_bins

    def train_calibration(self, zs, ys):
        self._sigmoid = utils.get_sigmoid(zs, ys)

    def calibrate(self, zs):
        return self._sigmoid(zs)

class ProbabilityCalibration():
    def _fit_multiclass(self, X, y, verbose=False):  
        """Fit the calibrated model in multiclass setting

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
    def _fit_multiclass(self, X, y, verbose=False):
        """Fit the calibrated model in multiclabel setting

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
def output():
    print("test package output!")