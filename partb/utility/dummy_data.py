import numpy as np
import pandas as pd


def random_dataset(n_features, n_instances):
    """
    Creates a random dataset with the given number of features and instances
    with binary class labels

    :param n_features: Number of features
    :param n_instances: Number of instances
    :return: matrix-like dataset (X) and list of target labels (y)

    :Example:
    >>> from partb.utility import dummy_data as dd

    >>> X, y = dd.random_dataset(6,15)
    """

    X = np.random.random((n_instances, n_features))
    y = np.random.randint(2, size=n_instances)#.tolist()

    return X, y


def random_dataframe(n_features, n_instances):
    """
    Creates a random dataframe with the given number of features and instances
    with binary target labels

    :param n_features: Number of features
    :param n_instances: Number of instances
    :return: pandas dataframe

    :Example:
    >>> from partb.utility import dummy_data as dd

    >>> rdf = dd.random_dataframe(6,15)
    >>> rdf.head()
    """

    X, y = random_dataset(n_features, n_instances)
    df = pd.DataFrame(X)
    df['target'] = y

    return df
