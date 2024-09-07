import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import dgl.backend as F
from dgl.data.utils import save_graphs
from dgllife.utils.io import pmap
from rdkit import Chem
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
from .featurizer import smiles_to_graph_tune
from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized

parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def onehot(labels,n_class):
    """print(onehot(np.array([[0],[1],[2]]),3))"""
    onehot = np.zeros((labels.shape[0], n_class))
    for i, value in enumerate(labels):
        onehot[i, value[0]] = 1
    return onehot

def preprocess_finetune_dataset(dataset='', n_tasks=2, dataset_dir=parent_parent_dir+'/dataset', path_length=5, n_jobs=24):
    df = pd.read_csv(f"{dataset_dir}/{dataset}/{dataset}.csv", sep='\t', header=0)
    cache_file_path = f"{dataset_dir}/{dataset}/{dataset}_{path_length}.pkl"
    smiless = df.Smiles.values.tolist()
    task_names = ['Label']
    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=path_length,
                  n_virtual_nodes=2,
                  n_jobs=n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    _label_values = df[task_names].values
    if n_tasks == 1:
        labels = F.zerocopy_from_numpy(_label_values.astype(np.float32))[valid_ids]
    else:
        _label_values = onehot(_label_values, n_tasks).astype(np.float32)
        labels = F.zerocopy_from_numpy(_label_values)[valid_ids]
    print('saving graphs')
    save_graphs(cache_file_path, valid_graphs,labels={'labels': labels})
                
    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{dataset_dir}/{dataset}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    np.savez_compressed(f"{dataset_dir}/{dataset}/molecular_descriptors.npz", md=arr[:, 1:])

if __name__ == '__main__':
    preprocess_finetune_dataset(dataset='')
