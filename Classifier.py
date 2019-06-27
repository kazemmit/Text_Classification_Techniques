from sklearn import metrics


def Classifier_Train_Test(classifier, feature_vector_train, label, feature_vector_test, test_y, is_neural_net=False):
    '''
    comments needed
    :param classifier:
    :param feature_vector_train:
    :param label:
    :param feature_vector_test:
    :param test_y:
    :param is_neural_net:
    :return:
    '''
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, test_y)