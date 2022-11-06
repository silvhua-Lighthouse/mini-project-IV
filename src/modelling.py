from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_grid_classifier(search_result, X_test, y_test, X_train, y_train, pos_label='1',model_name='Grid search'):
    """
    Print evaluation metrics of the best classifier estimator found from the grid search.
        * recall
        * precision
        * F1
        * AUC score (only if class labels are integers)
    Plot:
        * confusion matrix
        * ROC (only if class labels are integers)
    Parameters:
    - pos_label (str or int): Class label for positive class. Default is 1.
    - model_name (string, optional): Name of model printing purposes.

    Returns evaluation metrics for test data set as a dictionary
    """
    
    best_model = search_result.best_estimator_

    y_pred = best_model.predict(X_test)
    # print(y_pred)
    y_pred_train = best_model.predict(X_train)

    # Metrics for test data
    recall = recall_score(y_test, y_pred, pos_label=pos_label)
    precision = precision_score(y_test, y_pred, pos_label=pos_label)
    f1score = f1_score(y_test, y_pred, pos_label=pos_label)
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        'recall': recall,
        'precision': precision,
        'f1' : f1score,
        'accuracy': accuracy
    }

    # Metrics for training data
    recall_train = recall_score(y_train, y_pred_train, pos_label=pos_label)
    precision_train = precision_score(y_train, y_pred_train, pos_label=pos_label)
    f1score_train = f1_score(y_train, y_pred_train, pos_label=pos_label)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    print(f'\n{model_name} evaluation metrics: \n\tTest data\tTraining data\t\tDifference')
    print(f'Accuracy: \t{100*accuracy:.2f}%\t\t{100*accuracy_train:.2f}%\t\t{100*(accuracy-accuracy_train):.2f}%')
    print(f'Recall: \t{100*recall:.2f}%\t\t{100*recall_train:.2f}%\t\t{100*(recall-recall_train):.2f}%')
    print(f'Precision: \t{100*precision:.2f}%\t\t{100*precision_train:.2f}%\t\t{100*(precision-precision_train):.2f}%')
    print(f'F1: \t\t{100*f1score:.2f}%\t\t{100*f1score_train:.2f}%\t\t{100*(f1score-f1score_train):.2f}%')
    if type(y_test) == 'int':
        auc = roc_auc_score(y_test, y_pred)
        auc_train = roc_auc_score(y_train, y_pred_train)
        print(f'AUC: \t\t{100*auc:.2f}%\t\t{100*auc_train:.2f}%\t\t{100*(auc-auc_train):.2f}%')
        RocCurveDisplay.from_estimator(best_model, X_train, y_train)
        metrics['auc'] = auc
    
    print(f'Best model parameters from randomized search: {search_result.best_params_}')
    ConfusionMatrixDisplay.from_estimator(best_model, X_train, y_train)
    return metrics