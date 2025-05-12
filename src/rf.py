import pandas as pd
import numpy as np
from .trainer import BaseSklearnTrainer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from .data_loader import all_numpy_data_loader,kfolds_numpy_data_loader,load_scaffold
import joblib
import os
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))



RF_CLASSIFY_PARAM_GRID = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# RF_REGRESS_PARAM_GRID = {
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [3, 5, 7, 9, 12],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }


RF_REGRESS_PARAM_GRID = {
    'n_estimators': [50],
    'max_depth': [3],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}


def create_rf_regress_model(parameters={}):
    parameters['random_state'] = 42
    model = RandomForestRegressor(**parameters)
    return model

def create_rf_classify_model(parameters={}):
    parameters['random_state'] = 42 
    model = RandomForestClassifier(**parameters)
    return model

class RFSklearnTrainer(BaseSklearnTrainer):
    def __init__(self, create_clasify_model, create_regress_model,n_tasks=1):
        super(RFSklearnTrainer, self).__init__(create_clasify_model, create_regress_model, n_tasks)

    def _regress_param_grid(self):
        return RF_REGRESS_PARAM_GRID

    def _classify_param_grid(self):
        return RF_CLASSIFY_PARAM_GRID


def rf_predict(inputs, checkpoint_path, n_tasks):
    model = joblib.load(checkpoint_path)
    if n_tasks > 1 :
        return model.predict_proba(inputs)
    else:
        return np.expand_dims(model.predict(inputs), -1)


def rf_train(task_name, model_name, n_tasks, featurizer, text_type='Smiles', features_local=None):
    kfolds = kfolds_numpy_data_loader(task_name, n_tasks, featurizer=featurizer, text_type=text_type, features_local=features_local)
    all = all_numpy_data_loader(task_name, n_tasks, featurizer=featurizer, text_type=text_type, features_local=features_local)
    rf_trainer = RFSklearnTrainer(create_rf_classify_model,create_rf_regress_model,n_tasks=n_tasks)
    kf = load_scaffold(task_name)
    x,y = zip(*all)
    x,y = np.array(list(x)),np.array(list(y))
    rf_trainer.grid_search_paramers(kf, x, y, task_name, model_name)
    rf_trainer.kfold_cross_val(kfolds, task_name, model_name)
    # rf_trainer.load_best_params(task_name, model_name, root_path=parent_parent_dir+'/pretrained')
    rf_trainer.train_all(x, y, task_name, model_name)

