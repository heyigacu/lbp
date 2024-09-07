
import os 
import torch
import esm
import pandas as pd
import collections
import numpy as np
import os
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def esm_embeddings(peptide_sequence_list):
  # https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t6_8M_UR50D-contact-regression.pt
  model, alphabet = esm.pretrained.load_model_and_alphabet_local(parent_dir+'/esm2_t6_8M_UR50D.pt')
  batch_converter = alphabet.get_batch_converter()
  model.eval()  
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
  with torch.no_grad():
      results = model(batch_tokens, repr_layers=[6], return_contacts=True)  
  token_representations = results["representations"][6]
  sequence_representations = []
  for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
  embeddings_results = collections.defaultdict(list)
  for i in range(len(sequence_representations)):
      each_seq_rep = sequence_representations[i].tolist()
      for each_element in each_seq_rep:
          embeddings_results[i].append(each_element)
  embeddings_results = pd.DataFrame(embeddings_results).T
  return embeddings_results

def generate_esm_embeddings(tuple_ls, batch=500, save_path=''):
    tasks = list(range(0,len(tuple_ls),batch))
    for i in range(len(tasks)):
        print('task'+str(i))
        if i == 0:
            features = np.array(esm_embeddings(tuple_ls[tasks[i]:tasks[i+1]]))
        elif i != (len(tasks)-1):
            features = np.concatenate((features,np.array(esm_embeddings(tuple_ls[tasks[i]:tasks[i+1]]))),axis=0)
        else:
            features = np.concatenate((features,np.array(esm_embeddings(tuple_ls[tasks[i]:]))),axis=0)
    np.savetxt(save_path, features)

def generate_esm_embeddings_local(task_name, batch):
    df = pd.read_csv(parent_parent_dir+f'/dataset/{task_name}/{task_name}.csv', sep='\t', header=0)
    seq_ls = list(df['Sequence'])
    names = ['peptide' for i in seq_ls]
    tuple_ls = list(zip(names, seq_ls))
    save_path = parent_parent_dir+f'/dataset/{task_name}/esm_embeddings.txt'
    generate_esm_embeddings(tuple_ls, batch, save_path)