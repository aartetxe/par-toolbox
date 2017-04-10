import numpy as np
from sklearn.decomposition import PCA
import seaborn as sn
import matplotlib.pyplot as plt
from collections import OrderedDict


def histogram_plot(df, feature, target):
    """
    Plot histogram of a given feature

    :param df: Dataframe containing the dataset
    :param feature: Column name of the feature
    :param target: Column name of the target variable
    :return: The figure created
    """

    x1 = list(df[df[target] == 0][feature].values)
    x2 = list(df[df[target] == 1][feature].values)

    plt.figure() # TODO: Revise and remove if necessary

    plt.hist(x1, 10, stacked=True, normed=False)
    plt.hist(x2, 10, stacked=True, normed=False)

    plt.xlabel(feature)
    plt.ylabel('count')

    return plt.figure()


def plot_pairs(dataset, class_tag):
    """
    Plots pairwise relationships of a given dataset, taking into account the target label

    :param dataset:
    :param class_tag:
    :return:
    """
    sn.pairplot(dataset, hue=class_tag)

    sn.plt.show()
    return sn.plt.figure(1)


def plot_rf_k(k_results):  # TODO: Make it generic
    plt.plot(k_results[0], k_results[1])
    plt.xlabel('number of trees')
    plt.ylabel('accuracy')

    plt.show()
    return plt.figure(1)


def plot_classes(target, target_labels, save_file=""):
    unique, counts = np.unique(target, return_counts=True)
    plt.figure(figsize=(10, 10))
    sn.set_style("whitegrid")
    ax = sn.barplot(target_labels, counts)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x(), height + 3, '%s' % height)

    if save_file:
        plt.savefig("%s.png" % save_file)

    plt.show()
    return plt.figure(1)


def line_plot(dataset, target, neg_label='negative', pos_label='positive'):

    plt.figure(figsize=(10, 10))
    for index, row in dataset.iterrows():
        if target[index] == 0:
            plt.plot(row, label=neg_label, color='#A5E500', linewidth=1.0)
        else:
            plt.plot(row, label=pos_label, color='#3500BF', linewidth=1.0)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()
    return plt.figure()


def scatter_pca(dataset, target):
    colors = ['#A5E500', '#3500BF']
    # Convert to matrices
    X = dataset.as_matrix()
    y = target.as_matrix().astype(int)

    y_color = map(lambda x: colors[x], y)

    # Principal Component Analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_color)
    plt.title('Scatter PCA')
    plt.legend()
    plt.show()

    return plt.figure()


