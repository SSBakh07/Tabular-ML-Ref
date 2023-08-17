import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

seed = 7

class HistGBMModel():
    def __init__(self, x_train, y_train, cv_strat, metric='accuracy'):
        self.x_train, self.y_train = x_train, y_train
        self.metric = metric
        self.study = optuna.create_study(direction='maximize')
        self.trial = lambda trial : self.objective(trial, x_train, y_train, cv_strat)


    def objective(self, trial, X, y, cv_strat):
        params = {
            'max_iter': trial.suggest_int('max_iter', 50, 5000, 25),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.9, step=0.01),
            'max_bins': trial.suggest_int('max_bins', 2, 255, step=5),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.1, 5.0, step=0.1)
        }

        clf = HistGradientBoostingClassifier(random_state=seed, **params)

        scores = cross_validate(clf, X, y, cv=cv_strat, n_jobs=-1, scoring=self.metric)

        return scores['test_score'].mean()

    def run_trial(self, n_trials=100):
        self.study.optimize(self.trial, n_trials)

    def get_best_params(self):
        return self.study.best_params
    
    def get_best_model(self):
        best_params = self.get_best_params()
        clf = HistGradientBoostingClassifier(random_state=seed, **best_params)
        clf.fit(self.x_train, self.y_train)
        return clf