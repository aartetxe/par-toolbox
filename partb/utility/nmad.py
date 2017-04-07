from __future__ import division

import numpy as np

"""
NORMALIZED MEAN ABSOLUTE DEVIATION (NMAD)
"""


def nmad(X):
    """
    Function to calculate the normalized mean absolute deviation of a vector or ensemble of vectors.

    :param X: array of floats
    :return: normalized absolute deviation
    """
    data_mean = np.mean(X, axis=1)
    sub_data = (X.transpose() - data_mean).transpose()
    MAD = np.mean(np.absolute(sub_data), axis=1)
    data_max = np.amax(np.amax(X))
    data_min = np.amin(np.amin(X))
    NMAD = MAD / (data_max-data_min)

    return NMAD


def predict_nmad(X, threshold_factor=1.5):
    """
    Function that returns whether a given instance or set of instances
    are outliers according to normalized absolute deviation (NMAD)

    :param X: array of floats
    :param threshold_factor: threshold to consider the instance an outlier
    :return: boolean, whether the instance is an outlier
    """
    NMAD = nmad(X)
    devparam_threshold = np.mean(NMAD)+threshold_factor*np.std(NMAD)
    pred = []
    for i in range(len(NMAD)):
        cur = NMAD[i]
        if cur > devparam_threshold:
            pred.append(1)
        else:
            pred.append(0)

    return pred
