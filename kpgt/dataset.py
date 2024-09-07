
import os
import pandas as pd
import numpy as np
import argparse 
from copy import deepcopy
from rdkit import Chem
from multiprocessing import Pool
import scipy.sparse as sps

import torch
from torch.utils.data import Dataset
import dgl
import dgl.backend as F
from dgl.data.utils import load_graphs, save_graphs
from dgllife.utils.io import pmap
from .featurizer import smiles_to_graph, smiles_to_graph_tune
from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized

class FinetuneMoleculeDataset(Dataset):
    def __init__(self, root_path, dataset, n_tasks, path_length=5, n_virtual_nodes=2, split_name=None, split=None):
        SPLIT_TO_ID = {'train':0, 'val':1}
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/{dataset}_{path_length}.pkl")
        ecfp_path = os.path.join(root_path, f"{dataset}/rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, f"{dataset}/molecular_descriptors.npz")
        # Load Data
        df = pd.read_csv(dataset_path, header=0, sep='\t')
        print(len(df))
        if split is not None:
            split_path = os.path.join(root_path, f"{dataset}/scaffold/{split_name}.npy")
            use_idxs = np.load(split_path, allow_pickle=True)[SPLIT_TO_ID[split]]
        else: 
            split_path = os.path.join(root_path, f"{dataset}/{split_name}.npy")
            use_idxs = np.load(split_path, allow_pickle=True)
        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        self.df, self.fps, self.mds = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs]
        print(len(self.df))
        self.smiless = self.df['Smiles'].tolist()
        self.use_idxs = use_idxs
        # Dataset Setting
        self.task_names = self.df.columns.drop(['Smiles']).tolist()
        self.n_tasks = n_tasks
        self._pre_process()
        self.mean = None
        self.std = None
        if n_tasks > 1:
            self._task_pos_weights = self.task_pos_weights()
        elif n_tasks == 1:
            self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps,self.mds
    def __len__(self):
        return len(self.smiless)
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]
    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.labels,nan=0), axis=0)
        masks = F.zerocopy_from_numpy((~np.isnan(self.labels.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights
    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std

def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0],batch_num],axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)

class Collator_pretrain(object):
    def __init__(
        self, 
        vocab, 
        max_length, n_virtual_nodes, add_self_loop=True,
        candi_rate=0.15, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1,
        fp_disturb_rate=0.15, md_disturb_rate=0.15
        ):
        self.vocab = vocab
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate

        self.fp_disturb_rate = fp_disturb_rate
        self.md_disturb_rate = md_disturb_rate

    def bert_mask_nodes(self, g):
        n_nodes = g.number_of_nodes()
        all_ids = np.arange(0, n_nodes, 1, dtype=np.int64)
        valid_ids = torch.where(g.ndata['vavn']<=0)[0].numpy()
        valid_labels = g.ndata['label'][valid_ids].numpy()
        probs = np.ones(len(valid_labels))/len(valid_labels)
        unique_labels = np.unique(np.sort(valid_labels))
        for label in unique_labels:
            label_pos = (valid_labels==label)
            probs[label_pos] = probs[label_pos]/np.sum(label_pos)
        probs = probs/np.sum(probs)
        candi_ids = np.random.choice(valid_ids, size=int(len(valid_ids)*self.candi_rate),replace=False, p=probs)

        mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*self.mask_rate),replace=False)
        
        candi_ids = np.setdiff1d(candi_ids, mask_ids)
        replace_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*(self.replace_rate/(1-self.keep_rate))),replace=False)
        
        keep_ids = np.setdiff1d(candi_ids, replace_ids)

        g.ndata['mask'] = torch.zeros(n_nodes,dtype=torch.long)
        g.ndata['mask'][mask_ids] = 1
        g.ndata['mask'][replace_ids] = 2
        g.ndata['mask'][keep_ids] = 3
        sl_labels = g.ndata['label'][g.ndata['mask']>=1].clone()

        # Pre-replace
        new_ids = np.random.choice(valid_ids, size=len(replace_ids),replace=True, p=probs)
        replace_labels = g.ndata['label'][replace_ids].numpy()
        new_labels = g.ndata['label'][new_ids].numpy()
        is_equal = (replace_labels == new_labels)
        while(np.sum(is_equal)):
            new_ids[is_equal] = np.random.choice(valid_ids, size=np.sum(is_equal),replace=True, p=probs)
            new_labels = g.ndata['label'][new_ids].numpy()
            is_equal = (replace_labels == new_labels)
        g.ndata['begin_end'][replace_ids] = g.ndata['begin_end'][new_ids].clone()
        g.ndata['edge'][replace_ids] = g.ndata['edge'][new_ids].clone()
        g.ndata['vavn'][replace_ids] = g.ndata['vavn'][new_ids].clone()
        return sl_labels
    def disturb_fp(self, fp):
        fp = deepcopy(fp)
        b, d = fp.shape
        fp = fp.reshape(-1)
        disturb_ids = np.random.choice(b*d, int(b*d*self.fp_disturb_rate), replace=False)
        fp[disturb_ids] = 1 - fp[disturb_ids]
        return fp.reshape(b,d)
    def disturb_md(self, md):
        md = deepcopy(md)
        b, d = md.shape
        md = md.reshape(-1)
        sampled_ids = np.random.choice(b*d, int(b*d*self.md_disturb_rate), replace=False)
        a = torch.empty(len(sampled_ids)).uniform_(0, 1)
        sampled_md = a
        md[sampled_ids] = sampled_md
        return md.reshape(b,d)
    
    def __call__(self, samples):
        smiles_list, fps, mds = map(list, zip(*samples))
        graphs = []
        for smiles in smiles_list:
            graphs.append(smiles_to_graph(smiles, self.vocab, max_length=self.max_length, n_virtual_nodes=self.n_virtual_nodes, add_self_loop=self.add_self_loop))
        batched_graph = dgl.batch(graphs)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        sl_labels = self.bert_mask_nodes(batched_graph)
        disturbed_fps = self.disturb_fp(fps)
        disturbed_mds = self.disturb_md(mds)
        return smiles_list, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds

class Collator_tune(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
    def __call__(self, samples):
        smiles_list, graphs, fps, mds, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds, labels



class PredictMoleculeDataset(Dataset):
    def __init__(self, smiless, graphs, fps, mds, path_length=5, n_virtual_nodes=2):
        # Dataset Setting
        self.n_tasks = 1
        self.mean = None
        self.std = None
        self.fps = fps
        self.mds = mds
        # self._task_pos_weights = self.task_pos_weights()
        self.smiless = smiless
        self.graphs = graphs
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
    def __len__(self):
        return len(self.smiless)
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx]


class Collator_predict(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
    def __call__(self, samples):
        smiles_list, graphs, fps, mds = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds

