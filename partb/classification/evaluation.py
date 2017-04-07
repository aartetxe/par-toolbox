from __future__ import division
import numpy as np
from partb.classification import metrics as metr
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
#from imblearn.metrics import sensitivity_score
#from imblearn.metrics import specificity_score


def evaluate_model(classifier, X, y, resampler=None, reps=10, k=10, verbose=0):

    num_classes = len(np.unique(y))  # number of classes
    count = np.bincount(y)  # number of instances of each class

    majority_class = count.argmax()  # majority class
    majority_count = count.max()  # majority class
    minority_count = count.min()  # minority class
    
    imbalance_ratio = majority_count / minority_count
    
    print "\n** Summary **\n\nNumber of attributes: %s\n" \
          "Number of instances: %s\n" \
          "Number of clases: %s\n" \
          "Imbalance ratio (IR): %.2f (%s/%s)" % \
          (X.shape[1], X.shape[0], num_classes, imbalance_ratio, majority_count, minority_count)
    
    # Object to hold global results
    global_results = {
        'accuracy': [],
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'recall': []  # TODO: Check that equals to sensitivity
    }
    
    for i in range(reps):
        if verbose:
            print "Processing rep %s..." % i
        
        # Create an object to hold our results
        results = {
            'conf_matrix': np.empty([num_classes, num_classes]),
            'accuracy': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'recall': []  # TODO: Check that equals to sensitivity
        }

        kf = KFold(n_splits=k, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if resampler:
                X_res, y_res = resampler.fit_sample(X_train, y_train)  # Resample training data
                classifier.fit(X_res, y_res)  # Train classifier
            else:
                classifier.fit(X_train, y_train)  # Train classifier without resampling

            expected = y_test  # Actual labels
            predicted = classifier.predict(X_test)  # Predicted labels

            # Compute metrics
            conf_matrix = metrics.confusion_matrix(expected, predicted)
            results['conf_matrix'] += conf_matrix
            results['accuracy'].append(metrics.accuracy_score(expected, predicted))
            try:
                results['auc'].append(metrics.roc_auc_score(expected, predicted))
            except ValueError:
                print("Only one class present in predicted labels")

            if num_classes == 2:  # Binary classification
                #results['sensitivity'].append(metr.get_sensitivity(conf_matrix))
                #results['specificity'].append(metr.get_specificity(conf_matrix))
                results['sensitivity'].append(metr.sensitivity_score(expected, predicted))
                results['specificity'].append(metr.specificity_score(expected, predicted))
            else:
                results['precision'].append(metrics.precision_score(expected, predicted, average='macro'))
                results['recall'].append(metrics.recall_score(expected, predicted, average='macro'))

            if verbose:
                print "\nExecution #%s\n" \
                      "Confusion matrix:\n%s\n" \
                      "Accuracy:%s" %\
                      (i, results['conf_matrix'], np.mean(results['accuracy']))
                if num_classes == 2: # Binary classification
                    print "Sensitivity:%s\n" \
                          "Specificity:%s" % \
                          (np.mean(results['sensitivity']), np.mean(results['specificity']))
                else:
                    print "Precision:%s\n" \
                          "Recall:%s" % \
                          (np.mean(results['precision']), np.mean(results['recall']))

                print "AUC:%s\n" % np.mean(results['auc'])

        # Save results
        global_results['accuracy'].append(np.mean(results['accuracy']))
        global_results['sensitivity'].append(np.mean(results['sensitivity']))
        global_results['specificity'].append(np.mean(results['specificity']))
        global_results['precision'].append(np.mean(results['precision']))
        global_results['recall'].append(np.mean(results['recall']))
        global_results['auc'].append(np.mean(results['auc']))

    # Print final results
    print "\n** Results **\n\nMean accuracy: %.4f (sd=%.4f)" % \
          (np.mean(global_results['accuracy']), np.std(global_results['accuracy']))
    if num_classes == 2: # Binary classification
        print "Mean sensitivity: %.4f (sd=%.4f)" % \
              (np.mean(global_results['sensitivity']), np.std(global_results['sensitivity']))
        print "Mean specificity: %.4f (sd=%.4f)" % \
              (np.mean(global_results['specificity']), np.std(global_results['specificity']))
    else:
        print "Mean precision: %.4f (sd=%.4f)" % \
              (np.mean(global_results['precision']), np.std(global_results['precision']))
        print "Mean recall: %.4f (sd=%.4f)" % \
              (np.mean(global_results['recall']), np.std(global_results['recall']))
    print "Mean AUC: %.4f (sd=%.4f)" % \
          (np.mean(global_results['auc']), np.std(global_results['auc']))

    return global_results


# TODO: Only returns accuracy
def test_rf(X, y, no_tree_range=(1, 10)):
    """
    Returns an array containing metrics per number of trees
    """
    glob_res = []
    for num in range(no_tree_range[0], no_tree_range[1]):
        rf = RandomForestClassifier(n_estimators=num)
        # TODO: Evaluate classifier
        res = evaluate_model(rf, X, y, reps=1, k=3)
        glob_res.append(res['accuracy'][0])

    return range(no_tree_range[0], no_tree_range[1]), glob_res
