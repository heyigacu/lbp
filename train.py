import pandas as pd
from src.mlp import mlp_train
from src.weave_gnn import gnn_train
from src.feature import load_feature_local, Descriptors_peptide, MorganFingerPrint_smiles
from src.rf import rf_train
from src.svm import svm_train
from esm_embedding.feature import generate_esm_embeddings_local
from kpgt.main import kpgt_train
from kpgt.load_data import preprocess_finetune_dataset


def train(task_name, n_tasks, text_type):
    if text_type == 'Smiles':
        # preprocess_finetune_dataset(dataset=task_name, n_tasks=n_tasks)
        # kpgt_train(task_name=task_name, model_name='KPGT', n_tasks=n_tasks)
        mlp_train(task_name, model_name='MorganFP-MLP', n_tasks=n_tasks, model_params={'n_tasks':n_tasks, 'n_feats':1024}, text_type=text_type, lr=0.0005, featurizer=MorganFingerPrint_smiles, features_local=None)
        gnn_train(task_name, 'weaveGNN', n_tasks, model_params={'n_tasks':n_tasks}, lr=0.0005, text_type=text_type)
        # rf_train(task_name, 'MorganFP-RF', n_tasks, MorganFingerPrint_smiles, text_type=text_type)
        # svm_train(task_name, 'MorganFP-SVM', n_tasks, MorganFingerPrint_smiles, text_type=text_type)
        pass
    else:
        generate_esm_embeddings_local(task_name, 50)
        mlp_train(task_name, model_name='ESM-MLP', n_tasks=n_tasks, model_params={'n_tasks':n_tasks, 'n_feats':320}, lr=0.0005, featurizer=None, features_local=load_feature_local(task_name, 'esm_embeddings'))
        rf_train(task_name, 'ESM-RF', n_tasks, None, text_type=text_type, features_local=load_feature_local(task_name, 'esm_embeddings'))
        svm_train(task_name, 'ESM-SVM', n_tasks, None, text_type=text_type, features_local=load_feature_local(task_name, 'esm_embeddings'))
        pass

if __name__ == '__main__':
    # train('multitaste_mols', 7, 'Smiles')  # finished
    # train('antibacterial_mols', 2, 'Smiles') # finished
    # train('astringent_mols', 2, 'Smiles') # finished
    # train('multitaste_peps', 7, 'Sequence') # finished
    # train('antibacterial_peps', 2, 'Sequence') # finished
    # train('astringent_peps', 2, 'Sequence') # finished
    # train('astringent_mols_threshold', 1, 'Smiles')  # finished
    # train('astringent_peps_threshold', 1, 'Sequence')  # finished

    # train('afpeptide', 2, 'Sequence') # finished
    # train('afprotein', 2, 'Sequence') # finished
    
    # train('tox_brain', 2, 'Smiles') # finished
    # train('tox_bone', 2, 'Smiles') # finished
    # train('tox_pro', 2, 'Smiles') # finished