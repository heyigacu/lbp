

import os
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from esm_embedding.feature import esm_embeddings
from src.mlp import mlp_predict
from src.weave_gnn import gnn_predict
from src.data_loader import collate_predict_dgl_graphs
from src.feature import MorganFingerPrint_smiles,Graph_smiles
from src.rf import rf_predict
from src.svm import svm_predict
from kpgt.main import kpgt_predict
from preprocess_data import clean_error

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return e_x / e_x.sum(axis=-1, keepdims=True)

def esm_embedding_sequence(seq):
    tuple_ls = []
    tuple_ls.append(('predict',seq))
    features = np.squeeze(np.array(esm_embeddings(tuple_ls)), 0)
    return features

def logits2labels_probs(logits, classnames=['non-AFP', "AFP"]):
    label_ls = []
    argmaxs =  np.argmax(logits, axis=1)
    prob_ls = softmax(logits)
    for argmax in argmaxs:
        label_ls.append(classnames[argmax])
    return label_ls, prob_ls

def find_checkpoint_path(pretrained_root_path='./pretrained',task_name='antibacterial_mols',model_name='MorganFP-MLP'):
    model_checkpoint_dir = f'{pretrained_root_path}/{task_name}/{model_name}/'
    for file in os.listdir(model_checkpoint_dir):
        if file.startswith('all'):
            checkpoint_path = model_checkpoint_dir+file
    return checkpoint_path

def dict_logits(logits, class_names, right_ids, error_ids):
    # right_ids should is logits
    num_total = len(right_ids+error_ids)
    dic = {}
    if len(class_names) > 1:
        dic['Predict']=['Error Format' for _ in range(num_total)]
        for task in class_names:
            dic[task]=['0.' for _ in range(num_total)]
        labels, probs = logits2labels_probs(logits, class_names)
        for i,value in enumerate(right_ids):
            dic['Predict'][value]=labels[i]
            for j,task in enumerate(class_names):
                dic[task][value] = f"{probs[i][j]:.3f}"
        # for i,value in enumerate(error_ids):
        #     dic['Predict'][value]='Error Format'
    else:
        dic[class_names[0]]=['Error Format' for _ in range(num_total)]
        for i,value in enumerate(right_ids):
            dic[class_names[0]][value]=logits[i][0]
    return pd.DataFrame(dic)

def predict(input_path, output_path, task_name, model_name, text_type='Smiles', class_names=['threshold'], batch_size=500, 
            featurizer=Graph_smiles, n_feats=32, feature_local_path=None, pretrained_root_path='./pretrained', gnn_save_feats_dir=None):
    checkpoint_path = find_checkpoint_path(pretrained_root_path, task_name, model_name)
    df = pd.read_csv(input_path, sep='\t', header=0)
    clean_df, drop_list = clean_error(df, text_type)
    right_ids,error_ids=list(clean_df.index),drop_list
    texts = [text for text in list(clean_df[text_type])]
    if model_name == 'KPGT':
        logits = kpgt_predict(smiless=texts, n_tasks=len(class_names), model_name=model_name, task_name=task_name, batch_size=batch_size)
    elif model_name.endswith('RF'):
        if feature_local_path is not None:
             inputs = np.loadtxt(feature_local_path).astype(np.float32)
        else: inputs = np.array([featurizer(text) for text in texts]).astype(np.float32)
        logits = rf_predict(inputs, checkpoint_path, n_tasks=len(class_names))
    elif model_name.endswith('SVM'):
        if feature_local_path is not None:
             inputs = np.loadtxt(feature_local_path).astype(np.float32)
        else: inputs = np.array([featurizer(text) for text in texts]).astype(np.float32)
        logits = svm_predict(inputs, checkpoint_path, n_tasks=len(class_names))
    elif model_name.endswith('MLP'):
        if feature_local_path is not None:
             inputs = np.loadtxt(feature_local_path).astype(np.float32)
        else: inputs = np.array([featurizer(text) for text in texts]).astype(np.float32)
        print(inputs.shape)
        predict_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, collate_fn=None, drop_last=False)
        logits = mlp_predict(predict_loader, {'n_tasks':len(class_names), 'n_feats':n_feats}, checkpoint_path)
    elif model_name == 'weaveGNN':
        inputs = [featurizer(text, save_feats_dir=gnn_save_feats_dir) for text in texts]
        predict_loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_predict_dgl_graphs, drop_last=False)
        logits = gnn_predict(predict_loader, {'n_tasks':len(class_names)}, checkpoint_path, save_feats_dir=gnn_save_feats_dir)
    logits_df = dict_logits(logits,class_names,right_ids,error_ids)
    result_df = pd.concat([df, logits_df], axis=1)
    result_df.to_csv(output_path, index=False, header=True, sep='\t')
  
if __name__ == '__main__':
    # predict('dataset/multitaste_mols/multitaste_mols.csv', 'analysis/astringent_antibacterial_relation/multitaste_mols_KPGT_antibacterial_result.csv', 
    #         'antibacterial_mols', 'KPGT', text_type='Smiles', class_names=['Non-Antibacterial','Antibacterial'], batch_size=500,  featurizer=None, n_feats=None)
    
    # predict('analysis/astringent_antibacterial_relation/fda.csv', 'analysis/astringent_antibacterial_relation/fda_mols_KPGT_antibacterial_result.csv', 
    #     'antibacterial_mols', 'KPGT', text_type='Smiles', class_names=['Non-Antibacterial','Antibacterial'], batch_size=500,  featurizer=None, n_feats=None)
    # predict('analysis/astringent_antibacterial_relation/tcmsp.csv', 'analysis/astringent_antibacterial_relation/tcmsp_mols_KPGT_antibacterial_result.csv', 
    #     'antibacterial_mols', 'KPGT', text_type='Smiles', class_names=['Non-Antibacterial','Antibacterial'], batch_size=500,  featurizer=None, n_feats=None)

    # predict('analysis/astringent_antibacterial_relation/fda.csv', 'analysis/astringent_antibacterial_relation/fda_mols_KPGT_astringent_result.csv', 
    #     'astringent_mols', 'KPGT', text_type='Smiles', class_names=['Non-Astringent','Astringent'], batch_size=500,  featurizer=None, n_feats=None)
    predict('tanninc_acid.csv', 'output.csv', 
        'astringent_mols', 'KPGT', text_type='Smiles', class_names=['Non-Astringent','Astringent'], batch_size=500,  featurizer=None, n_feats=None)

    # predict('dataset/multitaste_peps/multitaste_peps.csv', 'analysis/astringent_antibacterial_relation/multitaste_peps_ESM-MLP_antibacterial_result.csv', 
    #         'antibacterial_peps', 'ESM-MLP', text_type='Sequence', class_names=['Non-Antibacterial','Antibacterial'], n_feats=320, batch_size=500, featurizer=esm_embedding_sequence)
    # predict('dataset/astringent_mols/astringent_mols.csv', 'analysis/astringent_antibacterial_relation/astringent_mols_MorganFP-MLP_antibacterial_result.csv', 
    #         'antibacterial_mols', 'MorganFP-MLP', text_type='Smiles', class_names=['Non-Antibacterial','Antibacterial'], batch_size=500,  featurizer=MorganFingerPrint_smiles, n_feats=1024)
    