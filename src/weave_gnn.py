
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  dgllife.model.gnn.weave import WeaveLayer
from  dgllife.model.readout import WeightedSumAndMax,SumAndMax
from  dgllife.model.readout import WeightedSumAndMax,SumAndMax
from .trainer import BaseTorchTrainer
from .feature import Graph_smiles
from .data_loader import kfolds_torch_data_loader,all_torch_data_loader, collate_train_dgl_graphs


parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class WeaveGNN(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=2,
                 hidden_feats=50,
                 activation=F.relu):
        super(WeaveGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(WeaveLayer(node_in_feats=node_in_feats,
                                                  edge_in_feats=edge_in_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))
            else:
                self.gnn_layers.append(WeaveLayer(node_in_feats=hidden_feats,
                                                  edge_in_feats=hidden_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))

    def reset_parameters(self):
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats, node_only=True, save_feats_dir=None):
        for i in range(len(self.gnn_layers)):
            if i == len(self.gnn_layers)-1 and node_only:
                node_feats = self.gnn_layers[-1](g, node_feats, edge_feats, node_only)
                if save_feats_dir is not None:
                    np.savetxt(f'{save_feats_dir}/layer{i+1}_node_feats.txt', node_feats.detach().numpy())
                return node_feats
            else:
                node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
                if save_feats_dir is not None:
                    np.savetxt(f'{save_feats_dir}/layer{i+1}_node_feats.txt', node_feats.detach().numpy())
                    np.savetxt(f'{save_feats_dir}/layer{i+1}_edge_feats.txt', edge_feats.detach().numpy())
        return node_feats,edge_feats
            
class WeavePredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=128,
                 gnn_activation=F.relu,
                 n_tasks=1):
        super(WeavePredictor, self).__init__()
        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)
        self.readout = WeightedSumAndMax(in_feats=gnn_hidden_feats)
        self.predict = nn.Sequential(
            nn.Linear(2*gnn_hidden_feats, 64),
            nn.Linear(64, n_tasks),
        )

    def forward(self, g, node_feats, edge_feats, save_feats_dir=None):
        node_feats,_ = self.gnn(g, node_feats, edge_feats, node_only=False, save_feats_dir=save_feats_dir)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)
    

def create_weave_gnn(n_tasks):
    model = WeavePredictor(node_in_feats=26, edge_in_feats=6, n_tasks=n_tasks)
    return model

def epoch_train_forward(train_loader, model, criterion, optimizer, device):
    model.train()
    for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
        graphs, labels = train_graphs.to(device), train_labels.to(device)
        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
        optimizer.zero_grad()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
    return model

def epoch_val_forward(val_loader, model, criterion, device):
    val_loss = 0.0
    current_preds = []
    current_labels = []
    with torch.no_grad():
        for val_graphs, val_labels in val_loader:
            graphs, labels = val_graphs.to(device), val_labels.to(device)
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            loss = criterion(preds, labels)
            val_loss += loss.item()
            current_preds.extend(preds.cpu().numpy())
            current_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, current_preds, current_labels


class GNNTorchTrainer(BaseTorchTrainer):
    def __init__(self, 
                 create_model, 
                 n_tasks=1,
                 model_params={}, 
                 lr=0.001, 
                 max_epochs=200, 
                 device='cuda'):
        super(GNNTorchTrainer, self).__init__(create_model, n_tasks, model_params, lr, max_epochs, device)

    def _epoch_train_forward(self):
        return epoch_train_forward

    def _epoch_val_forward(self):
        return epoch_val_forward
    

def gnn_predict(predict_loader, model_params={'n_tasks':2}, checkpoint_path='', save_feats_dir=None):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model=create_weave_gnn(**model_params)
    model.load_state_dict(state_dict)
    model.eval()
    total_preds = []
    with torch.no_grad():
        for graphs in predict_loader:
            outputs = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'), save_feats_dir)
            total_preds.extend(outputs.cpu().numpy())
    return np.array(total_preds)



def gnn_train(task_name, model_name, n_tasks, model_params={}, lr=0.001, text_type='Smiles'):
    kfolds = kfolds_torch_data_loader(task_name, n_tasks, batchsize=32, featurizer=Graph_smiles, collate_fn=collate_train_dgl_graphs, drop_last=False, text_type=text_type)
    gnn_trainer = GNNTorchTrainer(create_weave_gnn,n_tasks=n_tasks, model_params=model_params, lr=lr, max_epochs=500,device='cuda')
    gnn_trainer.kfold_cross_val(kfolds, task_name, model_name)
    # model_save_dir = f'{parent_parent_dir}/pretrained/{task_name}/{model_name}'
    # model_save_path = find_checkpoint_path(model_save_dir)
    # best_epoch = int(find_checkpoint_path(model_save_dir).split('/')[-1].split('.')[0][9:])
    # os.rename(model_save_path, model_save_path+'.old')
    # gnn_trainer.best_epoch = best_epoch
    
    # train_loader = all_torch_data_loader(task_name, n_tasks, batchsize=64, featurizer=Graph_smiles, collate_fn=collate_train_dgl_graphs, drop_last=False, text_type=text_type)
    # gnn_trainer.train_all(train_loader, task_name, model_name)



