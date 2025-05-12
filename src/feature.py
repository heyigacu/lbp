import torch
import numpy as np
import pandas as pd
import peptides
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from dgl import DGLGraph

import os
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), 
                                           [Chem.rdchem.HybridizationType.SP, 
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3, 
                                            Chem.rdchem.HybridizationType.SP3D])       
    return np.array(atom_features) 

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def Graph_smiles(smiles, save_feats_dir=None):
    molecule = Chem.MolFromSmiles(smiles)
    g = DGLGraph()
    g.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i) 
        node_features.append(atom_i_features)
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g.add_edges(i,j) 
                bond_features_ij = get_bond_features(bond_ij) 
                edge_features.append(bond_features_ij)
    if save_feats_dir is not None:
        np.savetxt(f'{save_feats_dir}/layer0_node_feats.txt', np.array(node_features))
        np.savetxt(f'{save_feats_dir}/layer0_edge_feats.txt', np.array(node_features))
    g.ndata['h'] = torch.from_numpy(np.array(node_features)).float()
    g.edata['e'] = torch.from_numpy(np.array(edge_features)).float()
    return g


def MorganFingerPrint_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

def Descriptors_peptide(sequence):
    """https://github.com/althonos/peptides.py dimension 88"""
    return np.array(list(peptides.Peptide(sequence).descriptors().values()))

def load_feature_local(task_name, embedding_name, root_path=parent_parent_dir+'/dataset'):
    return np.loadtxt(f'{root_path}/{task_name}/{embedding_name}.txt')


