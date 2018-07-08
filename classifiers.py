import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve

from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegressionCV
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

def process_data():
	application_train = pd.read_csv('data/application_train.csv', index_col=0)
	application_test = pd.read_csv('data/application_test.csv', index_col=0)

	full_set = pd.concat([application_train, application_test], sort=False)

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
	fold = KFold(len(y), n_folds=10, shuffle=True, random_state=777)
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

x_train, y_train, test = process_data()

clf, y_pred = lr_cv(x_train, y_train, test)

create_submissions(y_pred, 'submissions/LogisticRegressionCV.csv')








