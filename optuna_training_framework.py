import mlflow
import optuna
import numpy as np
import time
import random
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

TRACKING_SERVER = "http://mlflow.zorya-dynamics.cloud"

mlflow.set_tracking_uri(TRACKING_SERVER)

@dataclass
class DataParams:
    data: any
    labels: any
    random_state: int = random.randint(1, 1000000)
    test_size: int = 20

@dataclass
class OptunaParams:
    direction: str
    objective: callable
    storage: str = "sqlite:///my.db"
    

class OptunaRunner:
    def __init__(self, experiment_name: str, data_params: DataParams, optuna_params: OptunaParams):
        if experiment_name is None:
            print('Experiment name must be provided')
            raise
        if not isinstance(data_params, DataParams):
            print('DataParams are malformed')
            raise
        if not isinstance(optuna_params, OptunaParams):
            print('OptunaParams are malformed')
            raise

        mlflow.set_experiment(experiment_name)

        # FIXME: later to be replaced with load_study (remote pgdb)
        self.study = optuna.create_study(study_name=experiment_name, storage=optuna_params.storage, direction=optuna_params.direction, load_if_exists=True)

        self.objective = optuna_params.objective

        X_train, X_test, y_train, y_test = train_test_split(data_params.data, data_params.labels, test_size=data_params.test_size, stratify=data_params.labels, random_state=data_params.random_state)

        self.trial_data = {
            "X_train" : np.array(X_train),
            "X_test" : np.array(X_test),
            "y_train" : np.array(y_train),
            "y_test" : np.array(y_test),
        }
    
    def run_trial(self, n_trials):
        start_time = time.time()
        print("--- Trial starts ---")
        self.study.optimize(lambda trial: self.objective(trial, **self.trial_data), n_trials=n_trials)
        print("--- Trial took %s seconds ---" % (time.time() - start_time))
