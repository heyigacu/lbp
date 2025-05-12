import pandas as pd
import numpy as np
from .trainer import BaseSklearnTrainer
from sklearn.svm import SVC, SVR
from .data_loader import all_numpy_data_loader, kfolds_numpy_data_loader, load_scaffold
import joblib
import os
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

SVM_CLASSIFY_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

SVM_REGRESS_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

def create_svm_regress_model(parameters={}):
    # parameters['random_state'] = 42
    model = SVR(**parameters)
    return model

def create_svm_classify_model(parameters={}):
    parameters['random_state'] = 42
    parameters['probability'] = True
    model = SVC(**parameters)
    return model

class SVMSklearnTrainer(BaseSklearnTrainer):
    def __init__(self, create_classify_model, create_regress_model, n_tasks=1):
        super(SVMSklearnTrainer, self).__init__(create_classify_model, create_regress_model, n_tasks)

    def _regress_param_grid(self):
        return SVM_REGRESS_PARAM_GRID

    def _classify_param_grid(self):
        return SVM_CLASSIFY_PARAM_GRID

def svm_predict(inputs, checkpoint_path, n_tasks):
    model = joblib.load(checkpoint_path)
    if n_tasks > 1 :
        return model.predict_proba(inputs)
    else:
        return  np.expand_dims(model.predict(inputs), -1)


def svm_train(task_name, model_name, n_tasks, featurizer, text_type='Smiles', features_local=None):
    kfolds = kfolds_numpy_data_loader(task_name, n_tasks, featurizer=featurizer, text_type=text_type, features_local=features_local)
    all = all_numpy_data_loader(task_name, n_tasks, featurizer=featurizer, text_type=text_type, features_local=features_local)
    svm_trainer = SVMSklearnTrainer(create_svm_classify_model, create_svm_regress_model, n_tasks=n_tasks)
    kf = load_scaffold(task_name)
    x, y = zip(*all)
    x, y = np.array(list(x)), np.array(list(y))
    svm_trainer.grid_search_paramers(kf, x, y, task_name, model_name)
    svm_trainer.kfold_cross_val(kfolds, task_name, model_name)
    # svm_trainer.load_best_params(task_name, model_name, root_path=parent_parent_dir+'/pretrained')
    svm_trainer.train_all(x, y, task_name, model_name)

