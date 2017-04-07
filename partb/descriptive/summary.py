import numpy as np
from scipy import stats
from collections import Counter


def summary_table(df, continuous, target_label, output_format='html'):
    """
    This function generates a summary table of the given dataset.
    For the categorical variables the percentage is given,
    while for continuous data the arithmetic mean and standard deviation is shown

    :param df: Pandas dataframe containing the dataset
    :param continuous: array of column names of continuous variables
    :param target_label: column name of the target variable (string)
    :param (optional) output_format: format of the generated table, either html or latex (default='html')
    :return: summary table of the dataset (string)
    """
    groupby_readmitted = df.groupby(target_label)

    tnegs = groupby_readmitted.get_group(0.0).count()[1]
    tpos = groupby_readmitted.get_group(1.0).count()[1]
    tot = df.count()[1]

    outhtml = '<table>' \
              '<tr><th>Variable</th><th>All patients</th><th>Readmitted</th><th>Not readmitted</th><tr>'
    outhtml = outhtml + '<tr><td></td><td>(n=' + str(tot) + ')</td><td>(n=' + str(tpos) + ')</td></td><td>(n=' + str(
        tnegs) + ')</td></tr>'
    outtext = '\n' + ' ' + ' & ' + str(tot) + ' & ' + str(tpos) + ' & ' + str(tnegs) + ' & ' + " \\\\"

    for column in df:
        # Bypass target column
        if column == target_label:
            continue
        print column
        if column in continuous:  # Continuous variable
            name = column + ', mean (SD)'
            # Mean and standard deviation calculation
            # Total
            vals = df[column].as_matrix().astype(float)
            avg = "{0:.1f}".format(np.mean(vals))
            sd = "{0:.1f}".format(np.std(vals))
            total = avg + ' (' + sd + ')'
            # Positive
            arraypos = groupby_readmitted.get_group(1.0)[column]
            pavg = "{0:.1f}".format(arraypos.astype(float).mean())
            psd = "{0:.1f}".format(arraypos.astype(float).std())
            pos = pavg + ' (' + psd + ')'
            # Negative
            arrayneg = groupby_readmitted.get_group(0.0)[column]
            navg = "{0:.1f}".format(arrayneg.astype(float).mean())
            nsd = "{0:.1f}".format(arrayneg.astype(float).std())
            negs = navg + ' (' + nsd + ')'

            outhtml = outhtml + '<tr><td> - ' + name + '</td><td>' + total + '</td><td>' + pos + '</td></td><td>' + negs + '</td></tr>'
            outtext = outtext + '\n' + name + ' & ' + total + ' & ' + pos + ' & ' + negs + ' \\\\'

        else:  # Categorical variable
            colset = set(df[column])

            if len(colset) == 2:  # Binary categorical variable

                tot_items = len(df[column])
                num_items = len(df.loc[df[column] == 1])
                perc_items = "{0:.1f}".format(float(num_items) / float(tot_items) * 100)
                total = str(num_items) + ' (' + str(perc_items) + ')'

                arraypos = groupby_readmitted.get_group(1.0)[column]
                pos_num = len(arraypos.loc[arraypos == 1])
                pos_perc = "{0:.1f}".format(
                    float(pos_num) / float(tot_items) * 100)  # TODO: RESPECTO AL SUBCONJUNTO, NO EL TOTAL
                tpos = str(pos_num) + ' (' + str(pos_perc) + ')'

                arrayneg = groupby_readmitted.get_group(0.0)[column]
                neg_num = len(arrayneg.loc[arrayneg == 1])
                neg_perc = "{0:.1f}".format(float(neg_num) / float(tot_items) * 100)
                tneg = str(neg_num) + ' (' + str(neg_perc) + ')'

                outhtml = outhtml + '<tr><td> - ' + column + '</td><td>' + total + '</td><td>' + tpos + '</td></td><td>' + tneg + '</td></tr>'
                outtext = outtext + '\n' + column + ' & ' + total + ' & ' + tpos + ' & ' + tneg + ' \\\\'
            else:

                outhtml = outhtml + '<tr><td>' + column + ' (%)</td><td></td><td></td></td><td></td></tr>'
                outtext = outtext + '\n' + column + ' & ' + ' ' + ' & ' + ' ' + ' & ' + ' ' + ' \\\\'

                tot_items = len(df[column])
                for el in colset:
                    num_items = len(df.loc[df[column] == el])
                    perc_items = "{0:.1f}".format(float(num_items) / float(tot_items) * 100)
                    total = str(num_items) + ' (' + str(perc_items) + ')'

                    arraypos = groupby_readmitted.get_group(1.0)[column]
                    pos_num = len(arraypos.loc[arraypos == el])
                    pos_perc = "{0:.1f}".format(float(pos_num) / float(tot_items) * 100)
                    tpos = str(pos_num) + ' (' + str(pos_perc) + ')'

                    arrayneg = groupby_readmitted.get_group(0.0)[column]
                    neg_num = len(arrayneg.loc[arrayneg == el])
                    neg_perc = "{0:.1f}".format(float(neg_num) / float(tot_items) * 100)
                    tneg = str(neg_num) + ' (' + str(neg_perc) + ')'

                    outhtml = outhtml + '<tr><td> - ' + str(
                        el) + '</td><td>' + total + '</td><td>' + tpos + '</td></td><td>' + tneg + '</td></tr>'
                    outtext = outtext + '\n' + str(
                        el) + ' & ' + total + ' & ' + tpos + ' & ' + tneg + ' & ' + ' ' + ' \\\\'

    outhtml = outhtml + '</table>'

    if output_format == 'latex':
        return outtext
    elif output_format == 'html':
        return outhtml
    else:
        return 'ERROR: unknown output_format. Use \'html\' or \'latex\' instead'


def summary_table_pval(df, continuous, target_label, output_format='html'):
    """
        This function generates a summary table of the given dataset.
        For the categorical variables the percentage is given,
        while for continuous data the arithmetic mean and standard deviation is shown.

        Univariate statistics are also given (p-value):
        - Chi-square for categorical variables
        - t-test for continuous variables

        :param df: Pandas dataframe containing the dataset
        :param continuous: array of column names of continuous variables
        :param target_label: column name of the target variable (string)
        :param output_format: format of the generated table, either html or latex (default='html')
        :return: summary table of the dataset (string)
    """
    groupby_readmitted = df.groupby(target_label)

    tnegs = groupby_readmitted.get_group(0.0).count()[1]
    tpos = groupby_readmitted.get_group(1.0).count()[1]
    tot = df.count()[1]

    outhtml = '<table>' \
              '<tr><th>Variable</th><th>All patients</th><th>Readmitted</th><th>Not readmitted</th><th>p-value</th><tr>'
    outhtml = outhtml + '<tr><td></td><td>(n=' + str(tot) + ')</td><td>(n=' + str(tpos) + ')</td></td><td>(n=' + str(
        tnegs) + ')</td><td></td></tr>'
    outtext = '\n' + ' ' + ' & ' + str(tot) + ' & ' + str(tpos) + ' & ' + str(tnegs) + ' & ' + " \\\\"

    # Depending on the variable:
    # - Categorical -> Chi square
    # - Continuous -> t-test

    for column in df:
        # Bypass target column
        if column == target_label:
            continue
        print column
        if column in continuous:  # Continuous variable
            name = column + ', mean (SD)'
            # Mean and standard deviation calculation
            # Total
            vals = df[column].as_matrix().astype(float)
            avg = "{0:.1f}".format(np.mean(vals))
            sd = "{0:.1f}".format(np.std(vals))
            total = avg + ' (' + sd + ')'
            # Positive
            arraypos = groupby_readmitted.get_group(1.0)[column]
            pavg = "{0:.1f}".format(arraypos.astype(float).mean())
            psd = "{0:.1f}".format(arraypos.astype(float).std())
            pos = pavg + ' (' + psd + ')'
            # Negative
            arrayneg = groupby_readmitted.get_group(0.0)[column]
            navg = "{0:.1f}".format(arrayneg.astype(float).mean())
            nsd = "{0:.1f}".format(arrayneg.astype(float).std())
            negs = navg + ' (' + nsd + ')'
            # t-test
            t, p = stats.ttest_ind(arrayneg.as_matrix().astype(float), arraypos.as_matrix().astype(float),
                                   equal_var=False)
            # t, p = stats.ranksums(arrayneg.as_matrix().astype(float), arraypos.as_matrix().astype(float))
            cpval = "{0:.3f}".format(p)

            outhtml = outhtml + '<tr><td> - ' + name + '</td><td>' + total + '</td><td>' + pos + '</td></td><td>' + negs + '</td><td>' + cpval + '</td></tr>'
            outtext = outtext + '\n' + name + ' & ' + total + ' & ' + pos + ' & ' + negs + ' & ' + cpval + ' \\\\'

        else:  # Categorical variable
            colset = set(df[column])
            # chi-square
            nc = Counter(groupby_readmitted.get_group(0.0)[column].as_matrix())
            pc = Counter(groupby_readmitted.get_group(1.0)[column].as_matrix())
            a = []
            for oc in colset:
                a.append(nc[oc])
            b = []
            for oc in colset:
                b.append(pc[oc])
            obs = np.array([a, b])
            chi2, p, dof, exp = stats.chi2_contingency(obs)
            cpval = "{0:.3f}".format(p)

            if len(colset) == 2:  # Binary categorical variable

                tot_items = len(df[column])
                num_items = len(df.loc[df[column] == 1])
                perc_items = "{0:.1f}".format(float(num_items) / float(tot_items) * 100)
                total = str(num_items) + ' (' + str(perc_items) + ')'

                arraypos = groupby_readmitted.get_group(1.0)[column]
                pos_num = len(arraypos.loc[arraypos == 1])
                pos_perc = "{0:.1f}".format(
                    float(pos_num) / float(tot_items) * 100)  # TODO: RESPECTO AL SUBCONJUNTO, NO EL TOTAL
                tpos = str(pos_num) + ' (' + str(pos_perc) + ')'

                arrayneg = groupby_readmitted.get_group(0.0)[column]
                neg_num = len(arrayneg.loc[arrayneg == 1])
                neg_perc = "{0:.1f}".format(float(neg_num) / float(tot_items) * 100)
                tneg = str(neg_num) + ' (' + str(neg_perc) + ')'

                outhtml = outhtml + '<tr><td> - ' + column + '</td><td>' + total + '</td><td>' + tpos + '</td></td><td>' + tneg + '</td><td>' + cpval + '</td></tr>'
                outtext = outtext + '\n' + column + ' & ' + total + ' & ' + tpos + ' & ' + tneg + ' & ' + cpval + ' \\\\'
            else:

                outhtml = outhtml + '<tr><td>' + column + ' (%)</td><td></td><td></td></td><td></td><td>' + cpval + '</td></tr>'
                outtext = outtext + '\n' + column + ' & ' + ' ' + ' & ' + ' ' + ' & ' + ' ' + ' & ' + cpval + ' \\\\'

                tot_items = len(df[column])
                for el in colset:
                    num_items = len(df.loc[df[column] == el])
                    perc_items = "{0:.1f}".format(float(num_items) / float(tot_items) * 100)
                    total = str(num_items) + ' (' + str(perc_items) + ')'

                    arraypos = groupby_readmitted.get_group(1.0)[column]
                    pos_num = len(arraypos.loc[arraypos == el])
                    pos_perc = "{0:.1f}".format(float(pos_num) / float(tot_items) * 100)
                    tpos = str(pos_num) + ' (' + str(pos_perc) + ')'

                    arrayneg = groupby_readmitted.get_group(0.0)[column]
                    neg_num = len(arrayneg.loc[arrayneg == el])
                    neg_perc = "{0:.1f}".format(float(neg_num) / float(tot_items) * 100)
                    tneg = str(neg_num) + ' (' + str(neg_perc) + ')'

                    outhtml = outhtml + '<tr><td> - ' + str(
                        el) + '</td><td>' + total + '</td><td>' + tpos + '</td></td><td>' + tneg + '</td><td></td></tr>'
                    outtext = outtext + '\n' + str(
                        el) + ' & ' + total + ' & ' + tpos + ' & ' + tneg + ' & ' + ' ' + ' \\\\'

    outhtml = outhtml + '</table>'

    if output_format == 'latex':
        return outtext
    elif output_format == 'html':
        return outhtml
    else:
        return 'ERROR: unknown output_format. Use \'html\' or \'latex\' instead'

"""
# Usage in jupyter:

continuous = ['age', 'frecuencia_cardiaca', 'glucosa', 'temperatura']
outhtml = summary_table(df, continuous, 'readmitted')
display(HTML(outhtml))

"""