import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
import seaborn as sn
import matplotlib.pyplot as plt


def plot_cm(c_matrix, labels, save_file=""):
    """
    Plots a confusion matrix

    :param c_matrix: Array containing the confusion matrix itself
    :param labels: Array containing the ordered list of labels
    :param save_file: (optional) filename. By default saves nothing.
    :return: The figure created
    """

    array = c_matrix.astype(int)
    plt.figure(figsize=(10, 10))
    sn.heatmap(array, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    if save_file:
        plt.savefig("%s.png" % save_file)
    plt.show()
    return plt.figure(1)


def plot_roc(classifier, X, y, k=10, show_folds=True):
    """
    Plots Receiver Operating Curve for a given dataset and classifier

    :param classifier: Induction classifier implementing scikit's interfaces
    :param X: matrix-like dataset
    :param y: array of target labels
    :param k: (optional) number of folds in cross validation (default=10)
    :param show_folds: (optional) Plot each fold of cv (default True)
    :return: The figure created
    """

    kf = KFold(n_splits=k, shuffle=True)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # TODO: Consider including regular predictions (i.e. without probabilities)
        probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        if show_folds:
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random classifier')
    
    mean_tpr /= k
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return plt.figure(1)


def multiple_roc_plot(X, y, classifiers, names, k=5, title='Receiver operating characteristic'):
    """
    This method plots a comparison of ROC curves of a given set of classifiers

    :param X: matrix-like dataset
    :param y: array of target labels
    :param classifiers: array of classifiers implementing fit and predict_proba
    :param names: array of names of classifiers
    :param k: (optional) number of folds for cross validation (default=5)
    :param title: (optional) title of the generated figure
    :return: The figure created

    :Example:
    >>> from partb.utility import dummy_data as dd
    >>> from partb.visualization import visualization as viz
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = dd.random_dataset(8,150)

    >>> gb = GradientBoostingClassifier()
    >>> svm = SVC(kernel="linear", C=1, probability=True)
    >>> rf = RandomForestClassifier()

    >>> viz.multiple_roc_plot(X, y, classifiers=[gb, svm, rf], names=["GB", "SVM", "RF"])
    """

    linestyles = ['-', '.', '--', ':'] # TODO: If number of classifiers greater than linestyles, reuse cyclically

    kf = KFold(n_splits=k, shuffle=True)

    i = 0
    for classifier in classifiers:
        print names[i]

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # TODO: Consider including regular predictions (i.e. without probabilities)
            probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

        mean_tpr /= k
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, linestyles[i], label='%s (auc = %0.2f)' % (names[i], mean_auc), lw=2)
        i += 1

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random classifier')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    return plt.figure()


def plot_precision_recall_curve(classifier, X, y, k=10):  # TODO: Check interpolation, may not be correct
    """
    Plots Precision-Recall Curve for a given dataset and classifier

    :param classifier: Induction classifier implementing scikit's interfaces
    :param X: matrix-like dataset
    :param y: array of target labels
    :param k: (optional) number of folds in cross validation (default=10)
    :return: The figure created

    :Example:
    >>> from partb.utility import dummy_data as dd
    >>> from partb.visualization import visualization as viz
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> X, y = dd.random_dataset(8,150)
    >>> rf = RandomForestClassifier()
    >>> viz.plot_precision_recall_curve(rf, X, y)
    """

    kf = KFold(n_splits=k, shuffle=True)

    mean_recall = 0.0
    mean_precision = np.linspace(0, 1, 100)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])

        mean_recall += np.interp(mean_precision, precision, recall)
        #mean_recall[0] = 0.0

    mean_recall /= k
    #mean_recall[-1] = 1.0
    mean_auc = metrics.auc(mean_precision, mean_recall)
    plt.plot(mean_precision, mean_recall, 'k--', label='Mean auc = %0.2f' % mean_auc, lw=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(precision))
    plt.legend(loc="lower left")
    plt.show()

    return plt.figure()
