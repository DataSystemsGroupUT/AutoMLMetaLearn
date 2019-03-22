import itertools
import pandas as pd
import numpy as np
import sys

def permutations(dictionary):
    keys, values = zip(*dictionary.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(type(experiments))
    experiments = pd.DataFrame(experiments)

    return experiments


def generate_hyper_J48():
    np.random.seed(seed=0)

    C = np.unique(np.random.uniform(0, 1, 10))
    print("J48")

    param_dist_J48 = {

        # Do not collapse tree.
        "-O": [True, False],

        # Set minimum number of instances per leaf.
        "-M": [1, 2, 5, 8, 12, 15, 20, 30],

        # Use binary splits only.
        "-B": [True, False],

        #  Don't perform subtree raising.
        "-S": [True, False],

        # Do not clean up after the tree has been built.
        "-L": [True, False],

        # Laplace smoothing for predicted probabilities.
        "-A": [True, False],

        # Do not use MDL correction for info gain on numeric attributes.
        "-J": [True, False],

        # Seed for random data shuffling (default 1).
        "-Q": [1]
    }

    hyper_J48 = permutations(param_dist_J48)
    return hyper_J48


def generate_hyper_ABM1():
    print("ABM1")
    np.random.seed(seed=0)

    P = np.unique(np.random.randint(0, 101, 20))
    I = np.unique(np.random.randint(0, 201, 20))
    param_dist_ABM1 = {

        # Number of iterations.
        "-I": I,

        # Percentage of weight mass to base training on.
        "-P": P,

        # Use resampling for boosting.
        "-Q": [True, False],

        # Random number seed.
        "-S": [0]
    }

    hyper_ABM1 = permutations(param_dist_ABM1)

    return hyper_ABM1


# Logistic regression
def generate_hyper_LR():
    np.random.seed(seed=0)

    print("LR")
    iterations = np.unique(np.random.randint(0, 501, 20))
    iterations = np.append(iterations, -1)

    param_dist_LR = {
        # Set the maximum number of iterations (default -1, until convergence).
        "-M": iterations
    }

    hyper_LR = permutations(param_dist_LR)

    return hyper_LR


def generate_hyper_NB():
    np.random.seed(seed=0)

    print("NB")
    param_dist_NB = {
        # Use kernel density estimator rather than normal distribution for numeric attributes
        "-K": [True, False],

        #  Use supervised discretization to process numeric attributes
        "-D": [True, False],

        #  Display model in old format (good when there are many classes)
        "-O": [True, False],
    }

    hyper_NB = permutations(param_dist_NB)

    return hyper_NB


def generate_hyper_RF():

    print("RF")
    np.random.seed(seed=0)

    n_iter = np.unique(np.random.randint(5, 500, size=7)).tolist()
    batch_size = np.unique(np.random.randint(50, 500, size=7)).tolist()

    bag_size = np.unique(np.random.randint(0, 100, size=7)).tolist()
    variance = np.unique(np.random.uniform(-7, 5, size=7)).tolist()
    variance = np.exp(variance)

    param_dist_RF = {

        #  Size of each bag, as a percentage of the training set size. (default 100)
        "-P": bag_size,

        # Calculate the out of bag error.
        "-O": [True, False],

        #  Number of iterations.
        "-I": n_iter,

        #  Set minimum number of instances per leaf.
        "-M": [1, 2, 5, 8, 12, 15, 20, 30],

        #  Set minimum numeric class variance proportion of train variance for split (default 1e-3)
        "-V": variance,

        #  Seed for random number generator.
        "-S": [0],

        #  The maximum depth of the tree, 0 for unlimited.
        "-depth": [0, 10, 20, 30, 40, 50, 70],

        #  Number of folds for backfitting (default 0, no backfitting).
        "-N": [0, 10, 20, 30],

        #  Allow unclassified instances.
        "-U": [True, False],

        #  Break ties randomly when several attributes look equally good.
        "-B": [True, False],

        # The desired batch size for batch prediction
        "-batch-size": batch_size}

    hyper_RF = permutations(param_dist_RF)

    return hyper_RF


# REPTree
def generate_hyper_REP():
    print("REP")
    np.random.seed(seed=0)
    variance = np.unique(np.random.uniform(-7, 5, size=10)).tolist()
    variance = np.exp(variance)

    param_dist_REP = {
        #  Set minimum number of instances per leaf.
        "-M": [1, 2, 5, 8, 12, 15, 20, 30],

        #  Set minimum numeric class variance proportion of train variance for split (default 1e-3)
        "-V": variance,

        # Set number of folds for reduced error pruning. One fold is used as pruning set.
        "-N": [3, 5, 8, 12, 15],

        #  The maximum depth of the tree, -1 for unlimited.
        "-L": [0, 10, 20, 30, 40, 50, 70, 100],

        # Seed for random data shuffling (default 1).
        "-S": [0],

        #  No pruning.
        "-P": [True, False]
    }

    hyper_REP = permutations(param_dist_REP)

    return hyper_REP


def generate_hyper_PART():
    np.random.seed(seed=0)
    print("PART")

    param_dist_PART = {


        #  Generate unpruned decision list.
        "-U": [True, False],

        #  Set minimum number of objects per leaf.
        "-M": [1, 2, 5, 8, 12, 15, 20, 30],

        # Use binary splits only.
        "-B": [True, False],

        # Do not use MDL correction for info gain on numeric attributes.
        "-J": [True, False],

        # Seed for random data shuffling (default 1).
        "-Q": [0],

        # Do not make split point actual value.
        "-doNotMakeSplitPointActualValue": [True, False]}

    hyper_PART = permutations(param_dist_PART)

    return hyper_PART


# logistic model trees
def generate_hyper_LMT():
    np.random.seed(seed=0)
    print("LMT")
    beta = np.unique(np.random.uniform(-7, 5, size=10)).tolist()
    beta = np.exp(beta)
    min_split = np.unique(np.random.randint(2, 100, size=10)).tolist()
    I = np.unique(np.random.randint(0, 501, 10))

    param_dist_LMT = {
        #   Binary splits (convert nominal attributes to binary ones)

        "-B": [True, False],

        # Split on residuals instead of class values
        "-R": [True, False],

        #  Use error on probabilities instead of misclassification error for stopping criterion of LogitBoost.
        "-P": [True, False],

        #   Use cross-validation for boosting at all nodes (i.e., disable heuristic)
        "-C": [True, False],

        # The AIC is used to choose the best iteration.
        "-A": [True, False],

        #  Do not make split point actual value.
        "-doNotMakeSplitPointActualValue": [True, False],

        # Set beta for weight trimming for LogitBoost. Set to 0 (default) for no weight trimming.
        "-W": beta,

        # Set minimum number of instances at which a node can be split
        "-M": min_split,

        #  Set fixed number of iterations for LogitBoost (instead of using cross-validation)
        "-I": I

    }

    hyper_LMT = permutations(param_dist_LMT)

    return hyper_LMT


def generate_hyper_KStar():
    np.random.seed(seed=0)
    param_dist_KSTAR = {
        # Manual blend setting (default 20%)
        "-B": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

        #  Enable entropic auto-blend setting (symbolic class only)
        "-E": [True, False],

        #  Specify the missing value treatment mode (default a) Valid options are: a(verage), d(elete), m(axdiff), n(ormal)
        "-M": ["a", "d", "m", "n"]
    }

    hyper_KSTAR = permutations(param_dist_KSTAR)

    return hyper_KSTAR


# Using default search algorithm
def generate_hyper_IBk():
    np.random.seed(seed=0)
    print("IBk")
    param_dist_IBk = {
        # Weight neighbours by the inverse of their distance(use when k > 1)
        "-I": [True, False],

        # Weight neighbours by 1 - their distance (use when k > 1)
        "-F": [True, False],

        # Minimise mean squared error rather than mean absolute error when using -X option with numeric prediction. n(ormal)
        "-E": [True, False],

        #   Select the number of nearest neighbours between 1 and the k value specified using hold-one-out evaluation
        #   on the training data (use when k > 1)
        "-X": [True, False],
        #  Number of nearest neighbours (k) used in classification
        "-K": [1, 3, 7, 9, 13, 17, 19, 25, 29],
    }

    hyper_IBk = permutations(param_dist_IBk)

    return hyper_IBk


# HoeffdingTree
def generate_hyper_HF():
    print("HF")
    np.random.seed(seed=0)
    error = np.unique(np.random.uniform(-7, 5, size=10)).tolist()
    error = np.exp(error)
    threshold = np.unique(np.random.uniform(0, 5, size=10)).tolist()
    weight = np.unique(np.random.uniform(0, 1, size=10)).tolist()
    grace = np.unique(np.random.randint(0, 500, size=10)).tolist()
    inst = np.append(grace, 0).tolist()

    param_dist_HF = {

        #  The allowable error in a split decision - values closer to zero will take longer to decide
        "-E": error,

        #  Threshold below which a split will be forced to break ties
        "-H": threshold,

        # Minimum fraction of weight required down at least two branches for info gain splitting
        "-M": weight,

        # Grace period - the number of instances a leaf should observe between split attempts
        "-G": grace,

        # The number of instances (weight) a leaf should observe before allowing naive Bayes to make predictions (NB or NB adaptive only)
        "-N": inst,
    }

    hyper_HF = permutations(param_dist_HF)

    return hyper_HF


# Decision Table
# Default search method
def generate_hyper_DT():
    np.random.seed(seed=0)

    print("DT")
    param_dist_DT = {

        # Performance evaluation measure to use for selecting attributes.
        "-E": ["acc", "rmse", "mae", "auc"],

        #  Use nearest neighbour instead of global table majority n(ormal)
        "-I": [True, False],

        # Use cross validation to evaluate features.
        "-X": [1, 3, 5, 9, 13, 19, 25],

    }

    hyper_DT = permutations(param_dist_DT)

    return hyper_DT


def generate_hyper_OneR():
    print("OneR")
    np.random.seed(seed=0)

    param_dist_OneR = {
        # The minimum number of objects in a bucket
        "-B": [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

    }

    hyper_OneR = permutations(param_dist_OneR)

    return hyper_OneR


# With default kernel  and calibrator
def generate_hyper_SMO():
    np.random.seed(seed=0)
    print("SMO")
    epsilon = np.unique(np.random.uniform(-7, 0, size=10)).tolist()
    epsilon = np.exp(epsilon)

    tolerance = np.unique(np.random.uniform(-7, 5, size=10)).tolist()
    tolerance = np.exp(tolerance)
    param_dist_SMO = {
        #  Whether to 0=normalize/1=standardize/2=neither
        "-N": [0, 1, 2],
        # The complexity constant C.
        "-C": tolerance,
        # The tolerance parameter
        "-L": tolerance,
        # The epsilon for round-off error.
        "-P": epsilon,
        # Fit calibration models to SVM outputs.
        "-M": [True, False],
        #  The number of folds for the internal cross-validation. (default -1, use training data)
        "-V": [-1, 3, 5, 8, 12, 15, 20],
        # The random number seed.
        "-W": [0]
        # Options specific to weka.classifiers.functions.supportVector.PolyKernel:
        #
    }

    hyper_SMO = permutations(param_dist_SMO)

    return hyper_SMO


# Simple Logistic
def generate_hyper_SL():
    np.random.seed(seed=0)
    print("SL")
    iterations = np.unique(np.random.randint(0, 301, 20))
    beta = np.unique(np.random.uniform(-7, 5, size=10)).tolist()
    beta = np.exp(beta)
    beta = np.append(beta, 0)
    param_dist_SL = {
        # Set fixed number of iterations for LogitBoost
        "-I": iterations,
        #  Use stopping criterion on training set (instead of cross-validation)
        "-S": [True, False],
        #  Use error on probabilities (rmse) instead of misclassification error for stopping criterion
        "-P": [True, False],
        #  Set maximum number of boosting iterations
        "-M": iterations,
        #  Set parameter for heuristic for early stopping of  LogitBoost.If enabled, the minimum is selected greedily, stopping
        # if the current minimum has not changed for iter iterations. Set to zero to disable heuristic.
        "-H": [0, 20, 50, 70, 100],
        # Set beta for weight trimming for LogitBoost. Set to 0 for no weight trimming.
        "-W": beta,
        # The AIC is used to choose the best iteration (instead of CV or training error).
        "-A": [True, False]
    }
    hyper_SL = permutations(param_dist_SL)

    return hyper_SL


# Default classifier RePTree
def generate_hyper_Bagging():
    np.random.seed(seed=0)

    print("Bagging")
    iterations = np.unique(np.random.randint(0, 301, 20))
    param_dist_Bagging = {
        #  Full name of base classifier.
        "-W": ["weka.classifiers.trees.REPTree", "weka.classifiers.trees.J48"],
        #   Size of each bag, as a percentage of the training set size.
        "-P": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        #  Calculate the out of bag error.
        "-O": [True, False],
        # Random number seed.
        "-S": [0],
        #  Number of iterations.
        "-I": iterations, }

    hyper_Bagging = permutations(param_dist_Bagging)

    return hyper_Bagging


# Logit Boost
def generate_hyper_LB():
    np.random.seed(seed=0)

    print("LB")
    Z = np.unique(np.random.randint(0, 50, 10))
    improvement = np.unique(np.random.randint(0, 30, 20))
    improvement = -np.exp(improvement)
    improvement = np.append(improvement, -sys.float_info.max)
    iterations = np.unique(np.random.randint(0, 301, 20))

    shrinkage = np.unique(np.random.uniform(0, 1, 10))
    param_dist_LB = {
        # Use resampling instead of reweighting for boosting.
        "-Q": [True, False],

        #  Use estimated priors rather than uniform ones.
        "-use-estimated-priors": [True, False],
        # Percentage of weight mass to base training on.
        "-P": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # Random number seed
        "-S": [0],
        # Number of iterations.
        "-I": iterations,
        # Threshold on the improvement of the likelihood.
        "-L": improvement,
        # Z max threshold for responses.
        "-Z": Z,
        # Shrinkage parameter
        "-H": shrinkage,

    }

    hyper_LB = permutations(param_dist_LB)

    return hyper_LB