import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import os
from src.feature import Graph_smiles
from sklearn.model_selection import StratifiedKFold,KFold
from imblearn.over_sampling import RandomOverSampler

def check_rdkit(df_mol):
    drop_list = []
    for i,smiles in enumerate(list(df_mol['Smiles'])):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                drop_list.append(i)
        except:
            drop_list.append(i)
    print('drop {} unrecognized molecules by RDkit'.format(len(drop_list)))
    df_mol = df_mol.drop(drop_list)
    return df_mol, drop_list

def check_dgl(df_mol):
    drop_list = []
    for i,smiles in enumerate(list(df_mol['Smiles'])):
        try:
            Graph_smiles(smiles)
        except:
            drop_list.append(i)
    print('drop {} unknown molecules by DGL graph'.format(len(drop_list)))
    df_mol = df_mol.drop(drop_list)
    return df_mol, drop_list

def check_peptide(df_pep):
    legal_aas = 'ACDEFGHIKLMNPQRSTVWY'
    legal_aa_list = list(legal_aas)
    drop_list = []
    for i,row in df_pep.iterrows():
        for aa in list(row['Sequence']):
            if aa not in legal_aa_list:
                drop_list.append(i)
                break
            else:
                pass
    print('drop {} peptides with unknown amino acids'.format(len(drop_list)))
    df_pep = df_pep.drop(drop_list)
    return df_pep, drop_list

def add_canonical_smiles(df_mol):
    canonical_smiles_list = []
    drop_list = []
    for i,smiles in enumerate(df_mol['Smiles']):
        mol = Chem.MolFromSmiles(smiles)
        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            canonical_smiles_list.append(canonical_smiles)
        except:
            drop_list.append(i)
    print('drop {} unknown molecules by canonical SMILES'.format(len(drop_list)))
    df_mol = df_mol.drop(drop_list)
    df_mol.insert(len(df_mol.columns), 'CanonicalSmiles', canonical_smiles_list)
    return df_mol

def scaffold_kfold_split_classify(scaffold_dir,X,y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ros = RandomOverSampler(random_state=42)
    fold = 0
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
        resampled_indices = ros.sample_indices_
        resampled_train_indices = train_index[resampled_indices]
        arr = np.array([resampled_train_indices, test_index], dtype=object)
        np.save(scaffold_dir+'/scaffold-{}.npy'.format(fold), arr)
        fold += 1

def scaffold_kfold_split_regress(scaffold_dir,y):
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in skf.split(y):
        y_train = y[train_index]
        arr = np.array([train_index, test_index], dtype=object)
        np.save(scaffold_dir+'/scaffold-{}.npy'.format(fold), arr)
        fold += 1

def all_split_classify(datset_dir,X,y):
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X,y)
    resampled_indices = ros.sample_indices_
    np.save(datset_dir+'/all_sampled.npy', np.array(resampled_indices))


def clean_mols(task_name, n_tasks=1):
    df_mol = pd.read_csv(f'dataset/{task_name}/{task_name}.csv',header=0,sep='\t')
    # drop molecules unrecognized by RDkit
    df_mol,_ = check_rdkit(df_mol)
    # drop molecules unrecognized by DGL graph
    df_mol,_ = check_dgl(df_mol)
    if 'CanonicalSmiles' not in df_mol.columns:
        df_mol = add_canonical_smiles(df_mol)
    if n_tasks == 1:
        df_mol_sub_unique = df_mol.drop_duplicates(subset=['Taste', 'CanonicalSmiles']) # Label is threshold
    else:
        df_mol_sub_unique = df_mol.drop_duplicates(subset=['Label', 'CanonicalSmiles'])
    print('drop {} dulplicate of molecules for task {}'.format(len(df_mol)-len(df_mol_sub_unique),task_name))
    df_mol_sub_unique.to_csv(f'dataset/{task_name}/{task_name}.csv', header=True, index=False, sep='\t')


def clean_peps(task_name, n_tasks=1):
    df_pep = pd.read_csv(f'dataset/{task_name}/{task_name}.csv',header=0,sep='\t')
    # drop peps with unrecognized amino acids
    df_pep,_ = check_peptide(df_pep)
    if n_tasks == 1:
        df_pep_sub_unique = df_pep.drop_duplicates(subset=['Taste', 'Sequence']) # Label is threshold
    else:
        df_pep_sub_unique = df_pep.drop_duplicates(subset=['Label', 'Sequence'])
    print('drop {} dulplicate of peptides for task {}'.format(len(df_pep)-len(df_pep_sub_unique),task_name))
    df_pep_sub_unique.to_csv(f'dataset/{task_name}/{task_name}.csv', header=True, index=False, sep='\t')


def split_task(task_name, n_tasks=1):
    df = pd.read_csv('dataset/'+task_name+'/'+task_name+'.csv',header=0,sep='\t')
    if not os.path.exists('dataset/'+task_name+'/scaffold'):
        os.mkdir('dataset/'+task_name+'/scaffold')
    if n_tasks != 1:
        all_split_classify('dataset/'+task_name+'/', np.expand_dims(np.array(df['Label']), axis=1), np.array(df['Label']))
        scaffold_kfold_split_classify('dataset/'+task_name+'/scaffold',  np.expand_dims(np.array(df['Label']), axis=1), np.array(df['Label']))
    else:
        scaffold_kfold_split_regress('dataset/'+task_name+'/scaffold', np.array(df['Threshold']))

def preprocess_task(task_name, n_tasks=1, text_type='Smiles'):
    print(f'########## start process {task_name} ##########')
    if text_type == 'Smiles':
        # clean_error(task_name)
        clean_mols(task_name, n_tasks)
    else:
        clean_peps(task_name, n_tasks)
    split_task(task_name,  n_tasks)
    print(f'########## end process {task_name} ##########')


def clean_error(df, text_type):
    total_drop_list = []
    if text_type == 'Smiles':
        newdf,drop_list = check_rdkit(df)
        total_drop_list+=drop_list
        newdf,drop_list = check_dgl(newdf)
        total_drop_list+=drop_list
    elif text_type == 'Sequence':
        newdf,drop_list = check_peptide(df)
        total_drop_list+=drop_list
    return newdf,total_drop_list

if __name__ == '__main__': 
    # preprocess_task(task_name='multitaste_mols', n_tasks=8, text_type='Smiles')
    # preprocess_task(task_name='antibacterial_mols', n_tasks=2, text_type='Smiles')
    # preprocess_task(task_name='astringent_mols', n_tasks=2, text_type='Smiles')
    # preprocess_task(task_name='multitaste_peps', n_tasks=7, text_type='Sequence')
    # preprocess_task(task_name='antibacterial_peps', n_tasks=2, text_type='Sequence')
    # preprocess_task(task_name='astringent_peps', n_tasks=2, text_type='Sequence')
    # preprocess_task(task_name='astringent_mols_threshold', n_tasks=1, text_type='Smiles')
    # preprocess_task(task_name='astringent_peps_threshold', n_tasks=1, text_type='Sequence')
    # preprocess_task(task_name='afpeptide', n_tasks=2, text_type='Sequence')
    # preprocess_task(task_name='afprotein', n_tasks=2, text_type='Sequence')
    preprocess_task(task_name='tox_bone', n_tasks=2, text_type='Smiles')
    preprocess_task(task_name='tox_brain', n_tasks=2, text_type='Smiles')
    preprocess_task(task_name='tox_pro', n_tasks=2, text_type='Smiles')
