import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve

from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegressionCV

import xgboost as xgb
# TODO:
# Random forest
# - check feature importance

# Logistic Regression CV

# SVM CV


def plot_roc(y_test, y_pred_rt):
    fpr_rt, tpr_rt, _ = roc_curve(y_test, y_pred_rt)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt, tpr_rt, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


def read_in_data():
    application_train = pd.read_csv('data/application_train.csv', index_col=0)
    application_test = pd.read_csv('data/application_test.csv', index_col=0)
    return application_train, application_test


def impute(x):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imp.fit_transform(x)


def get_feature_columns(df):
    cols = df.columns.tolist()
    cols.remove('TARGET')
    return cols


def process_data(selected_cols='original'):
    application_train = pd.read_csv('data/application_train.csv', index_col=0)
    application_test = pd.read_csv('data/application_test.csv', index_col=0)

    full_set = pd.concat([application_train, application_test], sort=False).loc[:, selected_cols]

    full_set_with_dummies = pd.get_dummies(full_set, drop_first=True)

    feature_cols = full_set_with_dummies.columns.tolist()
    feature_cols.remove('TARGET')

    label1 = full_set_with_dummies.loc[full_set_with_dummies['TARGET'] == 1, :]
    label0 = full_set_with_dummies.loc[full_set_with_dummies['TARGET'] == 0, :]
    test = full_set_with_dummies.loc[full_set_with_dummies['TARGET'].isna(), feature_cols]

    label0_sample = label0.sample(label1.shape[0])

    balanced = pd.concat([label1, label0_sample], sort=False)
    x_train = balanced.loc[:, feature_cols]
    y_train = balanced.loc[:, 'TARGET']

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    x_train_imp = imp.fit_transform(x_train)
    test_imp = imp.fit_transform(test)

    return x_train_imp, y_train, test_imp


def lr_cv(x, y, test):
    fold = KFold(n_splits=10, shuffle=True, random_state=777)
    clf = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        ,penalty='l2'
        ,scoring='roc_auc'
        ,cv=fold
        ,random_state=777
        ,max_iter=10000
        ,fit_intercept=True
        ,solver='newton-cg'
        ,tol=10
    )
    clf.fit(x, y)
    print('Max roc_aucL', clf.scores_[1].max())
    y_pred = clf.predict(test)

    return clf, y_pred


def create_submissions(y_pred, submissions_name):
    submissions = pd.read_csv('data/sample_submission.csv')
    submissions.loc[:, 'TARGET'] = y_pred
    submissions.to_csv(submissions_name, index = False)


def feature_importance(x, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    plt.show()


def check_col_nas(df):
    cols = df.columns.tolist()
    percentage_nas = []
    for col in cols:
        percentage_nas.append(np.mean(df[col].isna()))
    return pd.DataFrame({
        'column_name': cols,
        'percentage_nas': percentage_nas
    })


def get_cols_for_dtype(df):
    unique_dtypes = np.unique(df.dtypes)
    dictionary = {}
    for d in unique_dtypes:
        cols = []
        for col in df.dtypes.index:
            if df.dtypes[col] == d:
                cols.append(col)
        dictionary[d] = cols
    return unique_dtypes, dictionary

# def get_col_types(df):
#     for col in df.columns.tolist():
#         if df[col]

# x_train, y_train, test = process_data()
#
# clf, y_pred = lr_cv(x_train, y_train, test)
#
# create_submissions(y_pred, 'submissions/LogisticRegressionCV.csv')








