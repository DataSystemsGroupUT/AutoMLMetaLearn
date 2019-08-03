import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
import glob
import warnings
import pandas as pd
from Weka.RandomizedSearch import generate_hyper_J48,generate_hyper_ABM1,generate_hyper_LR, generate_hyper_NB,generate_hyper_RF, generate_hyper_REP
from Weka.RandomizedSearch import generate_hyper_PART,generate_hyper_LMT, generate_hyper_KStar, generate_hyper_IBk, generate_hyper_HF
from Weka.RandomizedSearch import generate_hyper_DT,generate_hyper_OneR,generate_hyper_SMO, generate_hyper_SL,generate_hyper_Bagging,generate_hyper_LB
from Weka.Classification import classification



jvm.start(max_heap_size="15g")


classifiers_names = [
                     "J48", "AdaBoostM1", "LogisticRegression", "Naive Bayes",
                     "RandomForest", "REPTree", "PART", "Logistic Model Trees",
                     "KStar", "IBk", "HoeffdingTree", "Decision Table", "OneR",
                     "SMO", "Simple Logistic", "Bagging", "Logic Boost"
                    ]

columns = [
           "Dataset", "Classifier", "-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-I", "-J",
           "-K", "-L", "-M", "-N", "-O", "-P", "-Q", "-R", "-S", "-T", "-U", "-V","-X", "-Y",
           "-Z", "-doNotMakeSplitPointActualValue",
           "Train accuracy", "Train recall", "Train precision", "Train f1_score", "Train time",
           "Test accuracy", "Test recall", "Test precision", "Test f1_score", "Test time"
          ]

# List of 17 classifier functions implemented on Weka.
# It should follow same order as classifiers_names
classifier_functions = [Classifier(classname="weka.classifiers.trees.J48"),
                        Classifier(classname="weka.classifiers.meta.AdaBoostM1"),
                        Classifier(classname="weka.classifiers.functions.Logistic"),
                        Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
                        Classifier(classname="weka.classifiers.trees.RandomForest"),
                        Classifier(classname="weka.classifiers.trees.REPTree"),
                        Classifier(classname="weka.classifiers.rules.PART"),
                        Classifier(classname="weka.classifiers.trees.LMT"),
                        Classifier(classname="weka.classifiers.lazy.KStar"),
                        Classifier(classname="weka.classifiers.lazy.IBk"),
                        Classifier(classname="weka.classifiers.trees.HoeffdingTree"),
                        Classifier(classname="weka.classifiers.rules.DecisionTable"),
                        Classifier(classname="weka.classifiers.rules.OneR"),
                        Classifier(classname="weka.classifiers.functions.SMO"),
                        Classifier(classname="weka.classifiers.functions.SimpleLogistic"),
                        Classifier(classname="weka.classifiers.meta.Bagging"),
                        Classifier(classname="weka.classifiers.meta.LogitBoost")]




classifiers_logs = pd.DataFrame(columns=columns)

classifiers_logs.to_csv("ClassifierLogs.csv", index=False, columns=columns)

########################################################################################################################
### Step 1. Loop through all Datasets in the directory

path = 'Datasets_all'
allFiles = glob.glob(path + "/*.csv")
loader = Loader(classname="weka.core.converters.CSVLoader")
for file in allFiles:
    # Read the dataset into java instance format
    dataset = loader.load_file(file)


    # List of functions that run RandomizedSearchCV
    # The order should match with classifiers_names and classifiers_functions
    parameter_search_functions = [
                                   generate_hyper_J48(),     generate_hyper_ABM1(),
                                   generate_hyper_LR(),      generate_hyper_NB(),
                                   generate_hyper_RF(),      generate_hyper_REP(),
                                   generate_hyper_PART(),    generate_hyper_LMT(),
                                   generate_hyper_KStar(),   generate_hyper_IBk(),
                                   generate_hyper_HF(),      generate_hyper_DT(),
                                   generate_hyper_OneR(),    generate_hyper_SMO(),generate_hyper_SL(),
                                   generate_hyper_Bagging(), generate_hyper_LB()
                                 ]



    # Ignore warnings coming from unprocessed data
    warnings.filterwarnings("ignore")

########################################################################################################################
### Step 2. Loop through all classifiers stored in classifier functions.

    classification(file, dataset, classifiers_names, classifier_functions, parameter_search_functions, columns)

jvm.stop()