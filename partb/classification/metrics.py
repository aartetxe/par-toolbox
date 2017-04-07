from sklearn import metrics


def get_sensitivity(confusion):
    """
    sensitivity = TP / float(TP + FN)

    :param confusion: Confusion matrix
    :return: Sensitivity
    """

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    return TP / float(TP + FN)


def get_specificity(confusion):
    """
    specificity = TN / float(TN + FP)

    :param confusion: Confusion matrix
    :return: Specificity
    """

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    return TN / float(TN + FP)


def get_FNR(confusion):
    """
    FNR = FN / float(FN + TP)

    :param confusion: Confusion matrix
    :return: False Negative Rate or miss rate
    """

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    return FN / float(FN + TP)


def metric_FNR(actual, predicted):
    """

    :param actual: List of actual labels
    :param predicted: List of predicted labels
    :return: False Negative Rate or miss rate
    """
    confusion = metrics.confusion_matrix(actual, predicted)
    # TODO: Check size is 2x2
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    return FN / float(FN + TP)


def sensitivity_score(actual, predicted):
    """

    :param actual: List of actual labels
    :param predicted: List of predicted labels
    :return: Sensitivity
    """
    confusion = metrics.confusion_matrix(actual, predicted)
    c1, c2 = confusion.shape
    # TODO: Check size is 2x2
    if c1 != 2. or c2!=2.:
        print "malformed confusion matrix:\n%s" % confusion
        return 0

    TP = confusion[1, 1]
    FN = confusion[1, 0]

    return TP / float(TP + FN)


def specificity_score(actual, predicted):
    """

    :param actual: List of actual labels
    :param predicted: List of predicted labels
    :return: Specificity
    """
    confusion = metrics.confusion_matrix(actual, predicted)
    c1, c2 = confusion.shape
    # TODO: Check size is 2x2
    if c1 != 2. or c2 != 2.:
        print "malformed confusion matrix:\n%s" % confusion
        return 0

    TN = confusion[0, 0]
    FP = confusion[0, 1]

    return TN / float(TN + FP)
