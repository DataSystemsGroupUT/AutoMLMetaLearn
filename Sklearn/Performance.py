import pandas as pd
import glob
import warnings
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron, LogisticRegression
from Classification import  classification
from RandomizedSearchCv import generate_hyper_AB,generate_hyper_CNB,generate_hyper_DT,generate_hyper_GBC, generate_hyper_SVC
from RandomizedSearchCv import generate_hyper_GNB,generate_hyper_GP,generate_hyper_KNN,generate_hyper_LDA
from RandomizedSearchCv import generate_hyper_RF,generate_hyper_QDA,generate_hyper_LR,generate_hyper_Perceptron


# List of 12 classifier names


classifiers_names = [
                     "KNeighborsClassifier",
                     "GaussianProcessClassifier",
                     "DecisionTreeClassifier",
                     "RandomForestClassifier",
                     "AdaBoostClassifier",
                     "GaussianNB",
                     "QuadraticDiscriminantAnalysis",
                     "GradientBoostingClassifier",
                     "LinearDiscriminantAnalysis",
                     "Perceptron",
                     "LogisticRegression",
                     "ComplementNB",
                     "SVC"
                    ]

columns = [
           "Dataset", "Classifier", "n_neighbors", "weights", "algorithm", "leaf_size", "p", "metric","bootstrap",
           "C", "kernel", "degree", "gamma", "coef0", "shrinking", "shrinkage", "class_weight", "decision_function_shape",
           "random_state", "criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf",
           "oob_score", "warm_start", "learning_rate", "n_restarts_optimizer", "max_iter_predict", "max_features",
           "copy_X_train", "multi_class", "reg_param", "tol", "loss", "subsample", "min_weight_fraction_leaf", "n_estimators",
           "penalty", "dual", "fit_intercept", "intercept_scaling", "solver", "max_iter", "alpha", "shuffle", "max_leaf_nodes",
           "early_stopping", "n_iter_no_change", "fit_prior", "norm", "eta0", "var_smoothing", "presort", "min_impurity_decrease",
           "Train accuracy", "Train recall", "Train precision", "Train f1_score", "Train time", "Test accuracy",
           "Test recall", "Test precision", "Test f1_score", "Test time"
          ]

# List of 13 classifier functions implemented on scikit-learn.
# It should follow same order as classifiers_names and randomized_search_functions
classifier_functions = [
                        KNeighborsClassifier(),
                        GaussianProcessClassifier(),
                        DecisionTreeClassifier(),
                        RandomForestClassifier(),
                        AdaBoostClassifier(),
                        GaussianNB(),
                        QuadraticDiscriminantAnalysis(),
                        GradientBoostingClassifier(),
                        LinearDiscriminantAnalysis(),
                        Perceptron(),
                        LogisticRegression(),
                        ComplementNB(),
                        SVC()
                       ]


classifiers_logs = pd.DataFrame(columns=columns)

classifiers_logs.to_csv("ClassifierLogs.csv", index=False, columns=columns)

########################################################################################################################
### Step 1. Loop through all Datasets in the directory

path = 'Datasets_all'
allFiles = glob.glob(path + "/*.csv")

for file in allFiles:
    # Read the dataset into a pandas dataframe
    start = time.time()

    dataset = pd.read_csv(file, index_col=None, header=0)

    # List of functions that run RandomizedSearchCV
    # The order should match with classifiers_names and classifiers_functions
    randomized_search_functions = [
                                   generate_hyper_KNN(),   generate_hyper_GP(),
                                   generate_hyper_DT(),    generate_hyper_RF(),
                                   generate_hyper_AB(),    generate_hyper_GNB(),
                                   generate_hyper_QDA(),   generate_hyper_GBC(),
                                   generate_hyper_LDA(),   generate_hyper_Perceptron(),
                                   generate_hyper_LR(),    generate_hyper_CNB(),
                                   generate_hyper_SVC()
                                  ]

    # Ignore warnings coming from unprocessed data
    warnings.filterwarnings("ignore")

########################################################################################################################
### Step 2. Loop through all classifiers stored in classifier functions.

    classification(file, dataset, classifiers_names, classifier_functions, randomized_search_functions, columns)

    end = time.time()

    print(end-start, flush=True)
