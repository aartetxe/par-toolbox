import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


def anova(df, target_column, alpha=0.01):
    """
    Returns a dataframe containing only the features 
    with a p-value (ANOVA) lower than the given threshold
    
    :param df: 
    :param target_column: name of the target column
    :param alpha: p-value threshold
    :return: pandas dataframe containing the features 
    with a p-value lower than the given threshold
    """

    dataset = df.drop([target_column], axis=1)
    target = df[target_column]
    # Format dataset and label
    X = dataset.as_matrix()
    y = target.as_matrix().astype(int)

    selected_inds = []

    # ANalysis Of VAriance (ANOVA)
    selector = SelectPercentile(f_classif)
    selector.fit(X, y)

    print "Feature\t\tp-value\n============================\n"
    for i in range(len(selector.pvalues_)):
        print "%s\t\t %.4f" % (df.columns[i], selector.pvalues_[i])
        if selector.pvalues_[i] < alpha:
            selected_inds.append(i)

    sig_data = df.iloc[:,selected_inds]  # Filter significant variables
    sig_data = pd.concat([sig_data, df[target_column]], axis=1)

    return sig_data
