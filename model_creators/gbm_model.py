import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

seed = 7

class GBMModel():
    def __init__(self, x_train, y_train, cv_strat, metric='accuracy'):
        self.x_train, self.y_train = x_train, y_train
        self.metric = metric
        self.study = optuna.create_study(direction='maximize')
        self.trial = lambda trial : self.objective(trial, x_train, y_train, cv_strat)


    def objective(self, trial, X, y, cv_strat):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
            'n_estimators': trial.suggest_int('n_estimators', 50, 5000, 25),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),
        }

        clf = GradientBoostingClassifier(random_state=seed, **params)

        scores = cross_validate(clf, X, y, cv=cv_strat, n_jobs=-1, scoring=self.metric)

        return scores['test_score'].mean()

    def run_trial(self, n_trials=100):
        self.study.optimize(self.trial, n_trials)
    
    def get_best_params(self):
        return self.study.best_params
    
    def get_best_model(self):
        best_params = self.get_best_params()
        clf = GradientBoostingClassifier(random_state=seed, **best_params)
        clf.fit(self.x_train, self.y_train)
        return clf
