import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def generate_hyper_Perceptron():

    Percept = Perceptron()
    np.random.seed(seed=0)
    alpha = np.unique(np.random.uniform(-3, 12, 20)).tolist()
    alpha = np.exp(alpha)
    tol = alpha
    max_iter = np.unique(np.random.randint(5, 1000, size=20)).tolist()
    eta0 = alpha
    n_iter_no_change = np.unique(np.random.randint(5, 1000, size=20)).tolist()

    param_dist_Perceptron = {"penalty": ["l1", "l2", "elasticnet", None],
                             "alpha": alpha,
                             "fit_intercept": [True, False],
                             "max_iter": max_iter,
                             "tol": tol,
                             "shuffle": [True, False],
                             "eta0": eta0,
                             "early_stopping": [True, False],
                             "class_weight": [None, "balanced"],
                             "random_state": [0],
                             "warm_start": [True, False],
                             "n_iter_no_change": n_iter_no_change, }

    random_search = RandomizedSearchCV(Percept, param_distributions=param_dist_Perceptron,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=20, random_state=0, refit=False, n_jobs=-1)
    return random_search


def generate_hyper_KNN():
    knn = KNeighborsClassifier()
    param_dist_knn = {"n_neighbors": [1, 3, 7, 9, 13, 17, 19, 25, 29],
                      "weights": ["uniform", "distance"],
                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                      "leaf_size": [10, 30, 50, 70, 100],
                      "p": [1, 2],
                      "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"]}

    random_search = RandomizedSearchCV(knn, param_distributions=param_dist_knn,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=20, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_SVC():
    svm = SVC()

    np.random.seed(seed=0)
    distribution= np.unique(np.random.uniform(-3, 12, 20)).tolist()
    exponential=np.exp(distribution)


    param_dist_svm = {"C": exponential,
                      "kernel": ["rbf", "linear", "poly"],
                      "degree": [1, 2, 3, 4, 5],
                      "gamma": exponential,
                      "coef0": exponential,
                      "class_weight": ["balanced", None],
                      "random_state": [0]}
    random_search = RandomizedSearchCV(svm, param_distributions=param_dist_svm,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=10, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_GNB():

    gaussianNB = GaussianNB()

    np.random.seed(seed=0)
    var_smoothing = np.unique(np.random.uniform(-12, 0, 20)).tolist()
    var_smoothing = np.exp(var_smoothing)

    param_dist_GNB = {"var_smoothing": var_smoothing}

    random_search = RandomizedSearchCV(gaussianNB, param_dist_GNB,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=3, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_DT():
    DT = DecisionTreeClassifier()
    np.random.seed(seed=0)
    min_samples = np.unique(np.random.randint(2, 50, size=10)).tolist()
    max_leaf_nodes = np.unique(np.random.randint(10, 2000, size=10))
    max_leaf_nodes = np.append(max_leaf_nodes, None).tolist()
    min_impurity_decrease = np.unique(np.random.uniform(0, 1, 10))
    min_impurity_decrease = np.append(min_impurity_decrease, 0).tolist()

    param_dist_DT = {"criterion": ["gini", "entropy"],
                     "splitter": ["best", "random"],
                     "max_depth": [None, 10, 20, 30, 40, 50, 70, 100],
                     "min_samples_split": min_samples,
                     "min_samples_leaf": [1, 2, 5, 8, 12, 15, 20, 30],
                     "max_features": [None, "auto", "sqrt", "log2"],
                     "random_state": [0],
                     "max_leaf_nodes": max_leaf_nodes,
                     "min_impurity_decrease": min_impurity_decrease,
                     "class_weight": ["balanced", None],
                     "presort": [True, False]}

    random_search = RandomizedSearchCV(DT, param_distributions=param_dist_DT,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=20, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_RF():
    RF = RandomForestClassifier()
    np.random.seed(seed=0)

    n_estimators = np.unique(np.random.randint(10, 1000, size=20)).tolist()
    min_samples = np.unique(np.random.randint(2, 50, size=10)).tolist()
    max_leaf_nodes = np.unique(np.random.randint(10, 1000, size=50))
    max_leaf_nodes = np.append(max_leaf_nodes, None).tolist()

    min_impurity_decrease = np.unique(np.random.uniform(0, 100, 15))
    min_impurity_decrease = np.append(min_impurity_decrease, 0).tolist()

    param_dist_RF = {"n_estimators": n_estimators,
                     "criterion": ["gini", "entropy"],
                     "max_depth": [None, 10, 20, 30, 40, 50, 70, 100],
                     "min_samples_split": min_samples,
                     "min_samples_leaf": [1, 2, 5, 8, 12, 15, 20, 30],
                     "max_features": [None, "auto", "sqrt", "log2"],
                     "random_state": [0],
                     "max_leaf_nodes": max_leaf_nodes,
                     "min_impurity_decrease": min_impurity_decrease,
                     "bootstrap": [True, False],
                     "oob_score": [True, False],
                     "warm_start": [True, False],
                     "class_weight": ["balanced", "balanced_subsample", None]}

    random_search = RandomizedSearchCV(RF, param_distributions=param_dist_RF,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=30, random_state=0, refit=False, n_jobs=-1)
    return random_search


def generate_hyper_AB():
    AB = AdaBoostClassifier()

    np.random.seed(seed=0)

    n_estimators = np.unique(np.random.randint(10, 2000, size=30)).tolist()
    learning_rate = np.unique(np.random.uniform(-3, 12, size=10)).tolist()
    learning_rate=np.exp(learning_rate)

    param_dist_AB = {"random_state": [0],
                     "n_estimators": n_estimators,
                     "learning_rate": learning_rate,
                     "algorithm": ["SAMME", "SAMME.R"], }

    random_search = RandomizedSearchCV(AB, param_distributions=param_dist_AB,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=20, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_GP():
    GPC = GaussianProcessClassifier()

    np.random.seed(seed=0)
    max_iter_predict = np.unique(np.random.randint(0, 300, 10)).tolist()

    param_dist_GPC = {"n_restarts_optimizer": [0,  2,  5, 7, 12],
                      "max_iter_predict": max_iter_predict,
                      "warm_start": [True, False],
                      "copy_X_train": [True, False],
                      "random_state": [0],
                      "multi_class": ["one_vs_rest", "one_vs_one"]}

    random_search = RandomizedSearchCV(GPC, param_distributions=param_dist_GPC,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=10, random_state=0, refit=False, n_jobs=-1)

    return random_search


def generate_hyper_QDA():
    QDA = QuadraticDiscriminantAnalysis()

    np.random.seed(seed=0)
    reg_param = np.unique(np.random.uniform(-3, 12, 20)).tolist()
    reg_param = np.exp(reg_param)


    tol = np.unique(np.random.uniform(-3, 12, 20)).tolist()
    tol=np.exp(tol)

    param_dist_QDA = {"reg_param": reg_param,
                      "tol": tol}

    random_search = RandomizedSearchCV(QDA, param_distributions=param_dist_QDA,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=5, random_state=0, refit=False, n_jobs=-1)
    return random_search


def generate_hyper_GBC():
    GBC = GradientBoostingClassifier()

    np.random.seed(seed=0)
    learning_rate = np.unique(np.random.uniform(-5, 5, size=10)).tolist()
    learning_rate=np.exp(learning_rate)

    n_estimators = np.unique(np.random.randint(10, 1000, size=20)).tolist()

    subsample = np.unique(np.random.uniform(0, 1, size=10)).tolist()

    min_samples = np.unique(np.random.randint(2, 100, size=10)).tolist()
    min_weight = np.unique(np.random.uniform(0, 0.5, size=10))
    min_weight = np.append(min_weight, 0).tolist()

    min_impurity_decrease = np.unique(np.random.uniform(0, 1, 10))
    min_impurity_decrease = np.append(min_impurity_decrease, 0).tolist()
    max_leaf_nodes = np.unique(np.random.randint(10, 1000, size=10))
    max_leaf_nodes = np.append(max_leaf_nodes, None).tolist()
    n_iter_no_change = np.unique(np.random.randint(3, 15, size=10)).tolist()

    tol = np.unique(np.random.uniform(-3, 8, 20)).tolist()
    tol=np.exp(tol)

    param_dist_GBC = {"loss": ["deviance", "exponential"],
                      "learning_rate": learning_rate,
                      "n_estimators": n_estimators,
                      "subsample": subsample,
                      "criterion": ["friedman_mse", "mse", "mae"],
                      "min_samples_split": min_samples,
                      "min_samples_leaf": [1, 2, 5, 8, 12, 15, 20, 30],
                      "min_weight_fraction_leaf": min_weight,
                      "max_depth": [None, 10, 20, 30, 40, 50, 70, 100],
                      "min_impurity_decrease": min_impurity_decrease,
                      "random_state": [0],
                      "max_features": [None, "auto", "sqrt", "log2"],
                      "max_leaf_nodes": max_leaf_nodes,
                      "warm_start": [True, False],
                      "presort": [True, False, "auto"],
                      "n_iter_no_change": n_iter_no_change,
                      "tol": tol
                      }

    random_search = RandomizedSearchCV(GBC, param_distributions=param_dist_GBC,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=30, random_state=0, refit=False, n_jobs=-1)
    return random_search


def generate_hyper_LDA():
    LDA = LinearDiscriminantAnalysis()

    np.random.seed(seed=0)
    tol = np.unique(np.random.uniform(-3, 8, 20)).tolist()
    tol = np.exp(tol)

    param_dist_LDA = {"tol": tol,
                      "shrinkage": [None, "auto"],
                      "solver": ["svd", "lsqr", "eigen"]}

    random_search = RandomizedSearchCV(LDA, param_distributions=param_dist_LDA,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=5, random_state=0, refit=False,  n_jobs=-1)
    return random_search


def generate_hyper_LR():
    LR = LogisticRegression()

    np.random.seed(seed=0)


    distribution = np.unique(np.random.uniform(-3, 12, 20)).tolist()
    exponential = np.exp(distribution)

    intercept_scaling = np.unique(np.random.uniform(0, 20, 10)).tolist()
    max_iter = np.unique(np.random.randint(5, 1000, size=20)).tolist()

    param_dist_LR = {
                     "dual": [True, False],
                     "C": exponential,
                     "intercept_scaling": intercept_scaling,
                     "fit_intercept": [True, False],
                     "tol":exponential,
                     "class_weight": [None, "balanced"],
                     "random_state": [0],
                     "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                     "multi_class": ["ovr", "multinomial", "auto"],
                     "max_iter": max_iter,
                     "warm_start": [True, False],
                     }

    random_search = RandomizedSearchCV(LR, param_distributions=param_dist_LR,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=20, random_state=0, refit=False, n_jobs=-1)
    return random_search


def generate_hyper_CNB():
    CNB = ComplementNB()

    np.random.seed(seed=0)
    alpha = np.unique(np.random.uniform(-5, 12, 10)).tolist()
    alpha = np.exp(alpha)


    param_dist_CNB = {"alpha": alpha,
                      "fit_prior": [True, False],
                      "norm": [True, False]
                      }

    random_search = RandomizedSearchCV(CNB, param_distributions=param_dist_CNB,
                                       cv=2, error_score=np.nan, return_train_score=True,
                                       n_iter=5, random_state=0, refit=False, n_jobs=-1)
    return random_search
