
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import ast

parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def epoch_train_forward(train_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)  
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def epoch_val_forward(val_loader, model, criterion, device):
    val_loss = 0.0
    current_preds = []
    current_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            current_preds.extend(outputs.cpu().numpy())
            current_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, current_preds, current_labels

def save_predictions(predictions, labels, filepath):
    with open(filepath, 'w') as f:
        for prob, label in zip(predictions, labels):
            f.write(f'{prob.tolist()}, {label}\n')

def save_best_parameters(best_params, filepath):
    with open(filepath, 'w') as f:
        f.write(str(best_params))


def train_val(create_model, epoch_train_forward, epoch_val_forward, n_tasks=1, model_params={}, lr=0.001, max_epochs=200, patience=20, train_loader=None, val_loader=None, save_path_name='', device='cuda'):
    best_val_loss = float('inf')
    model = create_model(**model_params).to(device)
    if n_tasks == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_epoch = 0
    epochs_no_improve = 0
    for epoch in range(max_epochs):
        print(f'epoch {epoch+1}')
        model = epoch_train_forward(train_loader, model, criterion, optimizer, device)
        model.eval()
        val_loss, current_preds, current_labels = epoch_val_forward(val_loader, model, criterion, device)
        print(f'Epoch {epoch+1}/{max_epochs}, Validation Loss: {val_loss:.4f}, Best Loss: {best_val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_preds = current_preds
            best_labels = current_labels
            epochs_no_improve = 0 
            # torch.save(model.state_dict(), save_path_name+'.pth')
        else:
            epochs_no_improve += 1
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    best_epoch = best_epoch + 1
    save_predictions(best_preds, best_labels, save_path_name+str(best_epoch)+'.txt')
    return best_epoch


def kfold_cross_val(create_model, epoch_train_forward, epoch_val_forward, n_tasks=1,  model_params={}, lr=0.001,max_epochs=20, kfolds=None, task_name='', model_name='', device='cuda', root_path=parent_parent_dir+'/pretrained'):
    best_epochs = 0
    save_dir = f'{root_path}/{task_name}/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fold, (train_loader, val_loader) in enumerate(kfolds):
        best_epochs +=  train_val(create_model=create_model, epoch_train_forward=epoch_train_forward, epoch_val_forward=epoch_val_forward, n_tasks=n_tasks, model_params=model_params, lr=lr, max_epochs=max_epochs, train_loader=train_loader,val_loader=val_loader, save_path_name=f'{save_dir}/fold{fold}-epoch', device=device)
    return int(np.ceil(best_epochs/5))

def train_all(create_model, epoch_train_forward, n_tasks=1, model_params={}, lr=0.001, train_loader=None, max_epochs=10, task_name='', model_name='', device='cuda', root_path=parent_parent_dir+'/pretrained'):
    model = create_model(**model_params).to(device)
    if n_tasks == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max_epochs):
        print(f'epoch {epoch+1}')
        model = epoch_train_forward(train_loader, model, criterion, optimizer, device)
    save_name=f'all-epoch{max_epochs}'
    torch.save(model.state_dict(), f'{root_path}/{task_name}/{model_name}/{save_name}.pth')

class BaseTorchTrainer(object):
    def __init__(self, 
                 create_model, 
                 n_tasks=1, 
                 model_params={},
                 lr=0.001, 
                 max_epochs=200, 
                 device='cuda'):
        self.create_model = create_model
        self.n_tasks = n_tasks
        self.model_params = model_params
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = device
        self.epoch_train_forward = self._epoch_train_forward()
        self.epoch_val_forward = self._epoch_val_forward()
    
    def _epoch_train_forward(self):
        return epoch_train_forward

    def _epoch_val_forward(self):
        return epoch_val_forward

    def kfold_cross_val(self, kfolds=None, task_name='', model_name=''):
        self.best_epoch = kfold_cross_val(self.create_model, 
                                    self.epoch_train_forward, 
                                    self.epoch_val_forward, 
                                    self.n_tasks, 
                                    self.model_params,
                                    self.lr,
                                    self.max_epochs, 
                                    kfolds, 
                                    task_name, 
                                    model_name,
                                    self.device)
        

    def train_all(self, train_loader=None, task_name='', model_name=''):
        train_all(self.create_model, 
                  self.epoch_train_forward, 
                  self.n_tasks, 
                  self.model_params,
                  self.lr, 
                  train_loader,
                  self.best_epoch, 
                  task_name,
                  model_name,
                  self.device)

class BaseSklearnTrainer(object):
    def __init__(self, create_clasify_model, create_regress_model, n_tasks):
        if n_tasks == 1:
            self.create_model = create_regress_model 
            self.param_grid = self._regress_param_grid()
        else:
            self.create_model = create_clasify_model
            self.param_grid = self._classify_param_grid()
        self.n_tasks = n_tasks

    def _regress_param_grid():
        return {}
    
    def _classify_param_grid():
        return {}

    def grid_search_paramers(self, kf, x, y, task_name='', model_name=''):
        save_dir = parent_parent_dir+f'/pretrained/{task_name}/{model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        init_model = self.create_model()
        if self.n_tasks == 1:
            scoring = 'neg_mean_squared_error'
        elif self.n_tasks == 2:
            scoring = 'roc_auc'
        else:
            scoring = make_scorer(roc_auc_score, multi_class='ovr', average='micro', needs_proba=True)
        grid_search = GridSearchCV(estimator=init_model, param_grid=self.param_grid, cv=kf, scoring=scoring, n_jobs=-1, verbose=2)
        grid_search.fit(x, y)
        self.best_params = grid_search.best_params_
        save_best_parameters(self.best_params, save_dir+'/best_parameters.txt')
        return grid_search

    def kfold_cross_val(self, kfolds=None, task_name='', model_name=''):
        save_dir = parent_parent_dir+f'/pretrained/{task_name}/{model_name}'
        for i,(train,val) in enumerate(kfolds):
            x_train,y_train = tuple(zip(*train))
            x_train, y_train = np.array(list(x_train)), np.array(y_train)
            x_val,y_val = tuple(zip(*val))
            x_val,y_val = np.array(list(x_val)), np.array(y_val)
            model = self.create_model(self.best_params)
            model.fit(x_train, y_train)
            if self.n_tasks == 1:
                predicts = model.predict(x_val)
                predicts = np.expand_dims(predicts, -1)
            else:
                predicts = model.predict_proba(x_val)
            save_predictions(predicts, y_val, save_dir+f'/fold-{i}.txt')
            joblib.dump(model, save_dir+f'/fold-{i}.pkl')

    def load_best_params(self, task_name, model_name, root_path=parent_parent_dir+'/pretrained'):
        save_path = f'{root_path}/{task_name}/{model_name}/best_parameters.txt'
        dict_str = open(save_path,'r').readlines()[0].strip()
        self.best_params =  ast.literal_eval(dict_str)

    def train_all(self, x, y, task_name='', model_name='', root_path=parent_parent_dir+'/pretrained'):
        model = self.create_model(self.best_params)
        model.fit(x, y)
        joblib.dump(model, f'{root_path}/{task_name}/{model_name}/all.pkl')



