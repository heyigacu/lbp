import os
import random
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")

from .model import LiGhTPredictor as LiGhT
from .trainer import FinetuneTrainer
from .utils import set_random_seed, Result_Tracker, Evaluator, PolynomialDecayLR
from .featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES, smiles_to_graph_tune
from .dataset import FinetuneMoleculeDataset, preprocess_batch_light, Collator_tune, Collator_predict, PredictMoleculeDataset
from .load_data import preprocess_finetune_dataset

import dgl
import dgl.backend as F
from dgllife.utils.io import pmap

from rdkit import Chem
from scipy import sparse as sp
from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.data import Dataset

config_dict = {
    'base': {
        'd_node_feats': 137, 'd_edge_feats': 14, 'd_g_feats': 768, 'd_hpath_ratio': 12, 'n_mol_layers': 12, 'path_length': 5, 'n_heads': 12, 'n_ffn_dense_layers': 2,'input_drop':0.0, 'attn_drop': 0.1, 'feat_drop': 0.1, 'batch_size': 1024, 'lr': 2e-04, 'weight_decay': 1e-6,
        'candi_rate':0.5, 'fp_disturb_rate': 0.5, 'md_disturb_rate': 0.5
    }
}
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers-2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)

def finetune(dataset='predcoffee', n_tasks=2, model_name='kpgt', 
            seed=42, n_epochs=50, config_name='base', model_path=parent_parent_dir+'/pretrained/base_kpgt.pth', dataset_dir=parent_parent_dir+'/dataset',
            weight_decay=0., dropout=0, lr=3e-5, n_threads=8,):   
    set_random_seed(seed=seed)
    best_epochs = []
    model_save_dir = os.path.dirname(model_path)+f'/{dataset}/{model_name}/'
    config = config_dict[config_name]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for i in range(5):
        print('fold',i)
        model_save_path = model_save_dir + f'fold-{i}.pth'
        predictions_save_path = model_save_dir + f'fold{i}.txt'
        g = torch.Generator()
        g.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collator = Collator_tune(config['path_length'])
        train_dataset = FinetuneMoleculeDataset(root_path=dataset_dir, dataset=dataset, n_tasks=n_tasks, split_name=f'scaffold-{i}', split='train')
        val_dataset = FinetuneMoleculeDataset(root_path=dataset_dir, dataset=dataset, n_tasks=n_tasks, split_name=f'scaffold-{i}', split='val')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
        # Model Initialization
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=train_dataset.d_fps,
            d_md_feats=train_dataset.d_mds,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=dropout,
            feat_drop=dropout,
            n_node_types=vocab.vocab_size
        ).to(device)
        # Finetuning Setting
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(f'{model_path}').items()})
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
        print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6))
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=n_epochs*len(train_dataset)//32//10, tot_updates=n_epochs*len(train_dataset)//32,lr=lr, end_lr=1e-9,power=1)
        if n_tasks > 1:
            loss_fn = BCEWithLogitsLoss(reduction='none')
            metric='rocauc'
            evaluator = Evaluator(dataset, metric, train_dataset.n_tasks)
        else:
            loss_fn = MSELoss(reduction='none')
            metric='rmse'
            evaluator = Evaluator(dataset, metric, train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
        result_tracker = Result_Tracker(metric)
        summary_writer = None
        trainer = FinetuneTrainer(optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
        best_val, best_epoch = trainer.fit(model, train_loader, val_loader, val_dataset, model_save_path, predictions_save_path, n_epochs)
        best_epochs.append(int(best_epoch))
        print(f"val {metric}: {best_val:.3f}")
    # all train
    best_epoch = int(np.ceil(np.array(best_epochs).mean()))
    model_save_path = model_save_dir + f'all-epoch{best_epoch}.pth'
    g = torch.Generator()
    g.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])
    train_dataset = FinetuneMoleculeDataset(root_path=dataset_dir, dataset=dataset, n_tasks=n_tasks, split_name='all_sampled', split=None)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    # Model Initialization
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=dropout,
        feat_drop=dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    # Finetuning Setting
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(f'{model_path}').items()})
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=n_epochs*len(train_dataset)//32//10, tot_updates=n_epochs*len(train_dataset)//32,lr=lr, end_lr=1e-9,power=1)
    if n_tasks > 1:
        loss_fn = BCEWithLogitsLoss(reduction='none')
        metric='rocauc'
        evaluator = Evaluator(dataset, metric, train_dataset.n_tasks)
    else:
        loss_fn = MSELoss(reduction='none')
        metric='rmse'
        evaluator = Evaluator(dataset, metric, train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker(metric)
    summary_writer = None
    trainer = FinetuneTrainer(optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
    trainer.fit_all(model, train_loader, model_save_path, best_epoch)

def find_checkpoint_path(model_save_dir):
    for file in os.listdir(model_save_dir):
        if file.startswith('all'):
            checkpoint_path = model_save_dir+'/'+file
    return checkpoint_path

def predict(smiless=[], n_tasks=1, model_name='kpgt', dataset='', batch_size=100,
            seed=42, config_name='base', dataset_dir=parent_parent_dir+'/dataset', model_save_dir=parent_parent_dir+'/pretrained/',
            dropout=0, n_threads=8):
    set_random_seed(seed=seed)
    model_save_dir = model_save_dir+f'{dataset}/{model_name}/'
    model_save_path = find_checkpoint_path(model_save_dir)
    config = config_dict[config_name]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    if n_tasks == 1:
        train_dataset = FinetuneMoleculeDataset(root_path=dataset_dir, dataset=dataset, n_tasks=n_tasks, split_name=f'', split=None)
    g = torch.Generator()
    g.manual_seed(seed)
    device = torch.device("cpu")
    collator = Collator_predict(config['path_length'])

    print('constructing graphs')
    graphs = pmap(smiles_to_graph_tune, smiless, max_length=5, n_virtual_nodes=2, n_jobs=32)  
    valid_ids = []
    valid_graphs = []
    for i, g_ in enumerate(graphs):
        if g_ is not None:
            valid_ids.append(i)
            valid_graphs.append(g_)

    print('extracting fingerprints')
    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)
    fps = torch.from_numpy(FP_sp_mat.todense().astype(np.float32))

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    features_map = Pool(5).imap(generator.process, smiless)
    arr = np.array(list(features_map))
    md = arr[:,1:].astype(np.float32)
    mds = torch.from_numpy(np.where(np.isnan(md), 0, md))

    predict_dataset = PredictMoleculeDataset(smiless, graphs, fps, mds)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    
    # Model Initialization
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=predict_dataset.d_fps,
        d_md_feats=predict_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=dropout,
        feat_drop=dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    # Finetuning Setting
    model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=n_tasks, n_layers=2, predictor_drop=dropout, device=device, d_hidden_feats=256)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_save_path, map_location=device).items()})
    model.eval()
    ls = []
    for batched_data in predict_loader:
        (smiles, g, ecfp, md) = batched_data
        ecfp = ecfp.to(device)
        md = md.to(device)
        g = g.to(device)
        predictions = model.forward_tune(g, ecfp, md)
        preds = predictions.detach().cpu().numpy()
        print(preds)
        for pred in preds:
            if n_tasks == 1:
                ls.append(pred*train_dataset.std.numpy()[0]+train_dataset.mean.numpy()[0]) 
            else:
                ls.append(pred)
    return np.array(ls)



def kpgt_train(task_name='astringent_mols_threshold', model_name='KPGT', n_tasks=1):
    # preprocess_finetune_dataset(dataset=task_name, n_tasks=n_tasks)
    finetune(dataset=task_name, n_tasks=n_tasks, model_name=model_name)


def kpgt_predict(smiless=['CC(=C)C(CCC(=O)C)CC(=O)O','CCC1(C(=O)OCC)C(C)(c2ccccc2)O1'], task_name='astringent_mols_threshold', model_name='KPGT', n_tasks=1, batch_size=100):
    return predict(dataset=task_name, model_name=model_name, smiless=smiless, n_tasks=n_tasks, batch_size=batch_size)


    
    
