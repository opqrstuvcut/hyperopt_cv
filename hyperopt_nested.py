from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, space_eval
import numpy as np
import pandas as pd


class HyperOptCV(object):
    def __init__(self, estimator, space, scoring, inner_cv, outer_cv=None, test_size=None, evals=100, train_label=None, n_jobs=1):
        self.estimator = estimator
        self.space = space
        self.scoring = scoring
        self.evals = evals
        self.train_label = train_label
        self.n_jobs = n_jobs

        if outer_cv is None and test_size is None:
            raise ValueError("you must specify either outer_cv or test_size")

        if isinstance(outer_cv, int):
            self.outer_cv = KFold(
                n_splits=outer_cv, random_state=0, shuffle=True)
        elif outer_cv:
            self.outer_cv = outer_cv

        self.test_size = test_size

        if isinstance(inner_cv, int):
            self.inner_cv = KFold(
                n_splits=inner_cv, random_state=0, shuffle=True)
        else:
            self.inner_cv = inner_cv

    def _search_param_cv_core(self):
        trials = Trials()

        best = fmin(self.objective,
                    self.space,
                    algo=tpe.suggest,
                    max_evals=self.evals,
                    trials=trials)

        if isinstance(self.train_label, list):
            train_label_index = np.where(
                np.isin(self.train_y, self.train_label))[0]
            self.train_index = self.train_index[np.isin(
                self.train_index, train_label_index)]
            self.train_x, self.train_y = self.train_x[self.train_index], self.train_y[self.train_index]

        best = space_eval(self.space, best)
        self.estimator.set_params(**best)
        self.estimator.fit(self.train_x, self.train_y)

        self.scores.append(self.scoring(
            self.estimator, self.test_x, self.test_y))
        self.best_estimator_.append(best)
        self.val_scores.append(trials.best_trial["result"]["loss"])

    def search_param_cv(self, x, y):
        """search_param_cv
        訓練データとテストデータに分割し、訓練データでcvをおこなう。
        cvによって見つかった最良のモデルでテストデータの精度を計算して結果を返す。

        :param x:
        :param y:
        """
        self.scores = []
        self.best_estimator_ = []
        self.val_scores = []
        train_index, test_index = train_test_split(
            range(len(y)), test_size=self.test_size, random_state=42)
        if isinstance(x, pd.DataFrame):
            self.train_x = x.iloc[train_index, :]
            self.test_x = x.iloc[test_index, :]
        elif isinstance(x, np.ndarray):
            self.train_x = x[train_index]
            self.test_x = x[test_index]

        self.train_y = y[train_index]
        self.test_y = y[test_index]
        self.train_index = train_index

        self._search_param_cv_core()
        self.val_scores = np.array(self.val_scores)
        return np.array(self.scores)

    def search_param_outer_cv(self, x, y):
        self.scores = []
        self.best_estimator_ = []
        self.val_scores = []
        for train_index, test_index in self.outer_cv.split(x, y):
            if isinstance(x, pd.DataFrame):
                self.train_x = x.iloc[train_index, :]
                self.test_x = x.iloc[test_index, :]
            elif isinstance(x, np.ndarray):
                self.train_x = x[train_index]
                self.test_x = x[test_index]

            self.train_y = y[train_index]
            self.test_y = y[test_index]
            self.train_index = train_index

            self._search_param_cv_core()

        self.val_scores = np.array(self.val_scores)
        return np.array(self.scores)

    def objective(self, params):
        self.estimator.set_params(**params)
        score = cross_val_score(self.estimator, self.train_x, self.train_y,
                                cv=self.inner_cv, scoring=self.scoring, n_jobs=self.n_jobs)
        print(score.mean())
        return -score.mean()


class OneClassKFold(StratifiedKFold):
    def __init__(self, train_label, n_splits=3, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.train_label = train_label

    def split(self, x, y, group=None):
        train_label_index = np.where(np.isin(y, self.train_label))[0]
        for train_index, test_index in super().split(x, y):
            train_index = train_index[np.isin(train_index, train_label_index)]
            yield train_index, test_index


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn import svm
    from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer

    iris = load_iris()
    x = iris.data
    y = iris.target

    # classification
    space = {'C': hp.loguniform("C", np.log(1), np.log(100)),
             'kernel': hp.choice('kernel', ['rbf', 'poly']),
             'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1))}
    scoring = make_scorer(accuracy_score)
    hyperoptcv = HyperOptCV(estimator=svm.SVC(random_state=0), space=space, scoring=scoring,
                            outer_cv=StratifiedKFold(
                                n_splits=5, shuffle=True, random_state=0),
                            inner_cv=StratifiedKFold(
                                n_splits=10, shuffle=True, random_state=0),
                            test_size=0.2, evals=100)
    scores = hyperoptcv.search_param_outer_cv(x, y)
    print("test scores:{}".format(scores))
    print("best mean validation losses:{}".format(hyperoptcv.val_scores))

    scores = hyperoptcv.search_param_cv(x, y)
    print("test scores:{}".format(scores))
    print("best mean validation losses:{}".format(hyperoptcv.val_scores))

    # anomaly detection
    # x = x[y != 2]
    # y = y[y != 2]
    # y[y == 1] = -1
    # y[y == 0] = 1
    # space = {'kernel': hp.choice('kernel', ['rbf', 'poly']),
    #          'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1))}
    # # scoring = make_scorer(roc_auc_score)
    # scoring = make_scorer(accuracy_score)
    # ockfold = OneClassKFold(train_label=[1], n_splits=10, shuffle=True, random_state=0)
    # outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # hyperoptcv = HyperOptCV(estimator=svm.OneClassSVM(random_state=0), space=space, scoring=scoring,
    #                         outer_cv=outer_cv, inner_cv=ockfold, evals=100, train_label=[1])
    # scores = hyperoptcv.search_param_cv(x, y)
    # print("test scores:{}".format(scores))
    # print("validation losses:{}".format(hyperoptcv.val_scores))
