import pandas as pd
import time
import numpy as np
import numpy
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.filters import Filter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from func_timeout import func_timeout



def handle_numeric_pred(y_weka):
    """
    :param y_weka: Target values in java instance format
    :return: Target values in a python list

    """


    y_list=[]
    for i in y_weka:
        try:
            y_list.append(int(i))
        except Exception as error:
            print(error)
            y_list.append(999999)
    return y_list

def handle_string_pred(y_weka,dataset):

    """
    :param y_weka: Target values in java instance format
    :param dataset: Dataset in java instances format
    :return: Target values in a python list
    """
    y_list=[]
    for i in y_weka:
        try:
            ##Undo mapping between nominal values and integers
            y_list.append(dataset.class_attribute.value(int(i)))  ## What if it is a nominal-numeric target
        except Exception as error:
            print(error)
            y_list.append("abcdefgh")

    return y_list

def handle_numeric_extraction(y_weka):
    """
    :param y_weka: Target values in java instance format
    :return: Target values in a python list
    """
    y_list=[]
    for i in y_weka:
        try:
            y_list.append(int(i.values[0]))
        except Exception as error:
            print(error)
            y_list.append(999999)
    return y_list

def handle_string_extraction(y_weka,dataset):
    """
    :param y_weka: Target values in java instance format
    :param dataset: Dataset in java instances format
    :return: Target values in a python list
    """
    y_list=[]
    for i in y_weka:
        try:
            ##Undo mapping between nominal values and integers
            y_list.append(dataset.class_attribute.value(i.values[0]))
        except Exception as error:
            print(error)
            y_list.append("abcdefgh")

    return y_list


def classification(file, dataset, classifiers_names, classifier_functions, param_search_functions, columns):

    print(file, flush=True)
    class_index = dataset.num_attributes -1

    # class_name = dataset.attribute(class_index).name()

    # Determining which attribute is the class. Obligatory for Weka classifier
    dataset.class_is_last()

    numeric = False
    string = False



    # Weka classifiers treat numeric classes as a regression problem
    # Some classifiers can not deal with string labels
    # Convert class attribute to nominal.
    if dataset.attribute(class_index).is_numeric:
        numeric = True
        # Apply NumericToNominal filter to the last column which is our class
        convert =Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R" ,"last"])
        convert.inputformat(dataset)
        dataset =convert.filter(dataset)

    elif dataset.attribute(class_index).is_string:
        string =True

        # Apply StringToNominal filter to the last column which is our class
        convert =Filter(classname="weka.filters.unsupervised.attribute.StringToNominal", options=["-R" ,"last"])
        convert.inputformat(dataset)
        dataset =convert.filter(dataset)

    elif dataset.attribute(class_index).is_nominal:
        string =True

    train, test = dataset.train_test_split(80.0, Random(1))



    # Use the reverse option of remove filter to the last column
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last",  "-V"])
    # let the filter know about the type of data to filter
    remove.inputformat(train)

    # Save last column of train data to train_y
    train_y = remove.filter(train)
    remove.inputformat(test)

    # Save last column of test data to test_y
    test_y = remove.filter(test)

    # Create y_train and y_test list from java instances in order to be suitable to fit to metrics evaluators
    y_train = []
    y_test = []

    if string:
        y_train = handle_string_extraction(train_y, dataset)

        y_test = handle_string_extraction(test_y, dataset)

    elif numeric:
        y_train=handle_numeric_extraction(train_y)

        y_test=handle_numeric_extraction(test_y)

    i = 0
    for name in classifiers_names:

        print(name, flush=True)

        # Select the classifier to apply
        classifier = classifier_functions[i]

        # Select the hypermarameter generation function to apply in accordance with the classifier
        param_search = param_search_functions[i]

        log_size = param_search.shape[0]


        # Select at maximum 25 hyperparameter combinations
        if log_size >25:
            np.random.seed(seed=0)
            indices =np.random.choice(log_size, size=40, replace=False, p=None)
            param_search =param_search.iloc[indices,]
            log_size =25

        # Create a temporary empty logs dataframe
        logs = pd.DataFrame(index=range(0, log_size), columns=columns)
        i = i + 1



        #############################################################
        ### Run the classifier for every combination of parameters

        for j in range(log_size):
            logs.at[j, "Dataset"] = file

            ##Probably get the name from classifier parameter
            logs.at[j, "Classifier"] = "Weka. " +name

            # Store parameters generated in a list as an accepted format from classifier.
            param_values =[]


            for column in param_search.columns:

                # Store parameters in logs dataframe
                logs.at[j, column] = param_search.iloc[j][column]

                value = param_search.iloc[j][column]

                # Check if the hyperparameter is boolean in order to comply with weka parameters insertion syntax
                if isinstance(value, numpy.bool_):

                    # Just to avoid confusion when comparing 1 to True and 0 to False
                    if value==True:
                        # If boolean parameter is set to true append it to list of parameters
                        param_values.append(str(column))
                    else:
                        pass
                else:
                    # If not boolean append both parameter name and its value
                    param_values.append(str(column))
                    param_values.append(str(value))

            # Update the classifier parameters via options method
            try:
                classifier.options =param_values
            except (BaseException, Exception) as error:
                print(error, flush=True)
                continue

            # Build the classifier given train data
            try:
                start = time.perf_counter()
                classifier.build_classifier(train)
                end = time.perf_counter()

            except (BaseException, Exception) as error:
                print(error, flush=True)
                continue
            build_time = end - start

            # Fit the function and measure the time in fractional seconds
            try:
                start = time.perf_counter()
                evaluation = Evaluation(train)
                end = time.perf_counter()
            except (BaseException, Exception) as error:
                print(error, flush=True)
                continue

            eval_time = end - start
            logs.at[j, "Train time"] = eval_time + build_time

            # Predict labels for train data

            evl = evaluation.test_model(classifier, train) ### Predictions but in numeric formatl


            # Undo the mapping that Weka does from class labels to numbers
            if string:
                pred_train=handle_string_pred(evl,dataset)

            if numeric:
                pred_train=handle_numeric_pred(evl)


            # Predict labels for test data and measure time in fractional seconds
            start = time.perf_counter()
            evl = evaluation.test_model(classifier, test) ### Predictions but in numeric formatl
            end = time.perf_counter()
            test_time = end - start
            logs.at[j, "Test time"] = test_time

            # Undo the mapping that Weka does from class labels to numbers
            if string:
                pred_test=handle_string_pred(evl,dataset)

            if numeric:
                pred_test=handle_numeric_pred(evl)

            # Calculate train and test data accuracy and store the results in logs dataframe.
            train_accuracy = accuracy_score(y_train, pred_train)
            test_accuracy = accuracy_score(y_test, pred_test)
            logs.at[j, "Train accuracy"] = train_accuracy
            logs.at[j, "Test accuracy"] = test_accuracy

            # Calculate train and test data recall and store the results in logs dataframe.
            # For multiclass classification problems specify average method.
            # Calculate metrics globally by considering each element of the label indicator matrix as a label.
            train_recall = recall_score(y_train, pred_train, average="micro")
            test_recall = recall_score(y_test, pred_test, average="micro")
            logs.at[j, "Train recall"] = train_recall
            logs.at[j, "Test recall"] = test_recall

            # Calculate train and test data precision and store the results in logs dataframe.
            # For multiclass classification problems specify average method.
            # Calculate metrics globally by considering each element of the label indicator matrix as a label.
            train_precision = precision_score(y_train, pred_train, average="micro")
            test_precision = precision_score(y_test, pred_test, average="micro")
            logs.at[j, "Train precision"] = train_precision
            logs.at[j, "Test precision"] = test_precision

            # Calculate train and test data F1 score and store the results in logs dataframe.
            # For multiclass classification problems specify average method.
            # Calculate metrics globally by considering each element of the label indicator matrix as a label.
            train_f1_score = f1_score(y_train, pred_train, average="micro")
            test_f1_score = f1_score(y_test, pred_test, average="micro")
            logs.at[j, "Train f1_score"] = train_f1_score
            logs.at[j, "Test f1_score"] = test_f1_score

        # Write logs to file
        logs.to_csv("ClassifierLogs.csv", header=False, mode='a', columns=columns, index=False)

