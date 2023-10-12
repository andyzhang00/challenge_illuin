#!/usr/bin/env python3
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def map_class_to_int(class_names):
    """
    Return a dictionary that map each class to a unique integer and
    one other one that does the reverse mapping.
    Args:
        class_names : array containing all the class names
    Return:
        class_to_int : a dictionary mapping all the classes to an integer
    """
    class_to_label = dict()
    label_to_class = dict()
    for class_index, class_name in enumerate(class_names):
        class_to_label[class_name] = class_index
        label_to_class[class_index] = class_name
    return class_to_label, label_to_class


def convert_class_name_to_id(data, class_to_id_mapping):
    """
    This function transform the data using the mapping class_to_id_mapping
    Example : ["lost_luggage", "lost_luggage", "out_of_scope"] -> [0, 0, 1]
    Args:
        data : the list containing the different classes
        class_to_id_mapping : the mapping (dict) between the class id and class names
    Return:
        output : the data were all class names have been replaced by their id
    """
    for i in range(len(data)):
        data[i] = class_to_id_mapping[data[i]]
    output = list(data.astype(int))
    return output


def complete_eval(y_test, y_pred):
    """
    Return the evaluation of the prediction multiple metrics :
    accuracy, precision, recall, f1-score
    Args:
        y_test : true annotations
        y_pred : predictions
    Returns:
        accuracy, precision, recall and f1-score
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    fone_score = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, fone_score


def print_metrics(accuracy, precision, recall, f1_score):
    """
    This function print some usual metrics using two significant digits.
    Args:
        accuracy : The Accuracy we want to display
        precision : The Precision we want to display
        recall : The Recall we want to display
        f1_score : The F1-score we want to display
    Return:
    """
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score:.2f}")


def single_prediction(text, model, id2class, tfidf_vectorizer):
    """
    Performs a classification using model on the verbatim text
    Args:
        text : The verbatim we want to classify
        model : The classification model
        id2class : The dictionary containing the mapping between the class
                   index and the class name
        tfidf_vectorizer : The TfidfVectorizer used on the training data
                           of the model
    Returns :
        class_predicted : The name of the class predicted
    """
    tfidf_test = tfidf_vectorizer.transform([text])
    pred = model.predict(tfidf_test)
    class_predicted = id2class[pred[0]]
    return class_predicted


def display_basic_metrics(y_test, y_pred, class_names):
    # Evaluation for the complete test set
    accuracy, precision, recall, fone_score = complete_eval(y_test, y_pred)
    print_metrics(accuracy, precision, recall, fone_score)

    # Evaluation for every class in the test set
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)


def display_all_results(model, X_test, y_test, class_names):
    """
    Performs a complete evaluation of the model on the test set and display
    all the results
    The metrics used are : Accuracy, Precision, Recall, F1-score on the complete
    dataset but also for every classes. A confusion matrix is also displayed.
    Args :
        model : The classification model
        X_test : The test set
        y_test : The annotations of the test set
        class_names : The list of the class names classified by the classification model
    Returns :
    """
    y_pred = model.predict(X_test)
    display_basic_metrics(y_test, y_pred, class_names)
    ConfusionMatrixDisplay.from_estimator(model,
                                          X_test,
                                          y_test,
                                          display_labels=class_names,
                                          xticks_rotation='vertical')


def display_all_results_dnn(y_test, y_pred, class_names):
    """
    Performs a complete evaluation using the predictions on the test set and display
    all the results
    The metrics used are : Accuracy, Precision, Recall, F1-score on the complete
    dataset but also for every classes. A confusion matrix is also displayed.
    Args :
        model : The classification model
        X_test : The test set
        y_test : The annotations of the test set
        class_names : The list of the class names classified by the classification model
    Returns :
    """
    display_basic_metrics(y_test, y_pred, class_names)
    ConfusionMatrixDisplay.from_predictions(y_test,
                                            y_pred,
                                            display_labels=class_names,
                                            xticks_rotation='vertical')