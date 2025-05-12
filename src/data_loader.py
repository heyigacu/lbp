
import os
import numpy as np
import pandas as pd
import dgl
import torch
from .feature import Graph_smiles
from torch.utils.data.dataloader import DataLoader

parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def load_single_fold(task_name, fold, root_path=parent_parent_dir+'/dataset'):
    obj = np.load('{}/{}/scaffold/scaffold-{}.npy'.format(root_path, task_name, fold), allow_pickle=True)
    return obj[0], obj[1]

def load_scaffold(task_name, root_path=parent_parent_dir+'/dataset'):
    ls = []
    for fold in range(5):
        obj = np.load('{}/{}/scaffold/scaffold-{}.npy'.format(root_path, task_name, fold), allow_pickle=True)
        ls.append((obj[0], obj[1]))
    return ls

def load_all(task_name, root_path=parent_parent_dir+'/dataset'):
    obj = np.load('{}/{}/all_sampled.npy'.format(root_path, task_name), allow_pickle=True)
    return obj

def collate_train_dgl_graphs(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def collate_predict_dgl_graphs(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph


def kfolds_torch_data_loader(task_name, n_tasks, batchsize=32, featurizer=Graph_smiles, collate_fn=collate_train_dgl_graphs, drop_last=False, text_type='Smiles', features_local=None, root_path=parent_parent_dir+'/dataset'):
    df = pd.read_csv(f'{root_path}/{task_name}/{task_name}.csv', sep='\t', header=0)
    mol_inputs = list(df[text_type])
    if n_tasks == 1:
        labels = np.array(list(df['Label'])).astype(np.float32) 
    else:
        labels = np.array(list(df['Label'])).astype(np.int64)
    if features_local is None:
        features = [featurizer(mol_input) for mol_input in mol_inputs]
    else:
        features = features_local
    tuple_ls =  list(zip(features, labels))
    kfolds = []
    for i in range(5):
        train_idxs, val_idxs = load_single_fold(task_name, i)
        trains = [tuple_ls[index] for index in train_idxs]
        trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
        vals = [tuple_ls[index] for index in val_idxs]
        vals = DataLoader(vals,batch_size=len(vals), shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
        kfolds.append((trains,vals))
    return kfolds

def all_torch_data_loader(task_name, n_tasks, batchsize=32, featurizer=Graph_smiles, collate_fn=collate_train_dgl_graphs, drop_last=False, text_type='Smiles', features_local=None, root_path=parent_parent_dir+'/dataset'):
    df = pd.read_csv(f'{root_path}/{task_name}/{task_name}.csv', sep='\t', header=0)
    texts = list(df[text_type])
    if n_tasks == 1:
        labels = np.array(list(df['Label'])).astype(np.float32) 
    else:
        labels = np.array(list(df['Label'])).astype(np.int64)
    if features_local is None:
        features = [featurizer(text) for text in texts]
    else:
        features = features_local
    tuple_ls =  list(zip(features, labels))
    if n_tasks > 1:
        train_idxs = load_all(task_name)
        trains = [tuple_ls[index] for index in train_idxs]
    else:
        trains = tuple_ls
    print(f'origin:{len(tuple_ls)}--->sampled:{len(trains)}')
    train_loader = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
    return train_loader

def onehot(labels,n_class):
    """print(onehot(np.array([0,1,2]),3))"""
    onehot = np.zeros((labels.shape[-1], n_class))
    for i, value in enumerate(labels):
        onehot[i, value] = 1
    return onehot

def kfolds_numpy_data_loader(task_name, n_tasks, featurizer=Graph_smiles, text_type='Smiles', features_local=None, root_path=parent_parent_dir+'/dataset'):
    df = pd.read_csv(f'{root_path}/{task_name}/{task_name}.csv', sep='\t', header=0)
    texts = list(df[text_type])
    if features_local is None:
        features = [featurizer(text) for text in texts]
    else:
        features = features_local
    labels = list(df['Label'])
    tuple_ls =  list(zip(features, labels))
    kfolds = []
    for i in range(5):
        train_idxs, val_idxs = load_single_fold(task_name, i)
        trains = [tuple_ls[index] for index in train_idxs]
        vals = [tuple_ls[index] for index in val_idxs]
        kfolds.append((trains,vals))
    return kfolds

def all_numpy_data_loader(task_name, n_tasks, featurizer=Graph_smiles,  text_type='Smiles', features_local=None, root_path=parent_parent_dir+'/dataset'):
    df = pd.read_csv(f'{root_path}/{task_name}/{task_name}.csv', sep='\t', header=0)
    texts = list(df[text_type])
    if features_local is None:
        features = [featurizer(text) for text in texts]
    else:
        features = features_local
    labels = list(df['Label'])
    tuple_ls =  list(zip(features, labels))
    if n_tasks > 1:
        train_idxs = load_all(task_name)
        trains = [tuple_ls[index] for index in train_idxs]
    else:
        trains = tuple_ls
    print(f'origin:{len(tuple_ls)}--->sampled:{len(trains)}')
    return trains


