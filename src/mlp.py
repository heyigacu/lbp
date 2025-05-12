import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import BaseTorchTrainer
from .data_loader import kfolds_torch_data_loader,all_torch_data_loader
from .feature import MorganFingerPrint_smiles, load_feature_local, Descriptors_peptide
import numpy as np
import os
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class MLP(nn.Module):
    def __init__(self, n_feats, n_tasks):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_tasks),
        )
    def forward(self, x):
        x = self.model(x)
        return x

def create_mlp(n_tasks, n_feats=320):
    model = MLP(n_feats=n_feats, n_tasks=n_tasks)
    return model

def mlp_predict(predict_loader, model_params={'n_tasks':2, 'n_feats':1024},  checkpoint_path=''):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model=create_mlp(**model_params)
    model.load_state_dict(state_dict)
    model.eval()
    total_preds = []
    with torch.no_grad():
        for inputs in predict_loader:
            outputs = model(inputs.float())
            total_preds.extend(outputs.cpu().numpy())
    return np.array(total_preds)

def mlp_train(task_name, model_name, n_tasks, model_params={}, text_type='Sequence', lr=0.001, featurizer=Descriptors_peptide, collate_fn=None, features_local=None):
    kfolds = kfolds_torch_data_loader(task_name, n_tasks, batchsize=64, featurizer=featurizer, collate_fn=collate_fn, drop_last=False, text_type=text_type, features_local=features_local)
    mlp_trainer = BaseTorchTrainer(create_mlp, n_tasks=n_tasks, model_params=model_params, lr=lr, max_epochs=500,device='cuda')
    mlp_trainer.kfold_cross_val(kfolds, task_name, model_name)
    
    # model_save_dir = f'{parent_parent_dir}/pretrained/{task_name}/{model_name}'
    # model_save_path = find_checkpoint_path(model_save_dir)
    # best_epoch = int(find_checkpoint_path(model_save_dir).split('/')[-1].split('.')[0][9:])
    # os.rename(model_save_path, model_save_path+'.old')
    # mlp_trainer.best_epoch = best_epoch

    train_loader = all_torch_data_loader(task_name, n_tasks, batchsize=64, featurizer=featurizer, collate_fn=collate_fn, drop_last=False, text_type=text_type, features_local=features_local)
    mlp_trainer.train_all(train_loader, task_name, model_name)

