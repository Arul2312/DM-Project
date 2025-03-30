import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from sklearn.cluster import DBSCAN
from Bio import PDB
from Bio.PDB.PDBList import PDBList
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Seq import Seq
from Bio import pairwise2

def load_data():
    pharmacologically_active = pd.read_csv('../../Datasets/raw/pharmacologically_active.csv')
    target_labels = pd.read_csv('../../Datasets/raw/target_labels.csv')
    dti = pd.read_csv('../../Datasets/raw/dti_dataset.csv')
    drugbank = pd.read_csv('../../Datasets/raw/drugbank.csv')
    protein_sequences = pd.read_csv('../../Datasets/raw/protein_sequences.csv')
    conf = pd.read_csv('../../Datasets/raw/confirmed_interactions.csv')

    return pharmacologically_active, target_labels, dti, drugbank, protein_sequences, conf

def compute_drug_similarity(smiles1, smiles2, morgan_gen):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = morgan_gen.GetFingerprint(mol1)
    fp2 = morgan_gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def construct_similarity_matrices(drugbank, pharmacologically_active, proteins):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    target_labels = pd.read_csv('../../Datasets/raw/target_labels.csv')
    drugs = list(target_labels.ID)

    drugbank_filtered = drugbank[drugbank['Drug id'].isin(drugs)]
    drugbank_filtered.index = np.arange(0, drugbank_filtered.shape[0])

    n_drugs = drugbank_filtered.shape[0]

    #D_D_1
    D_D_1 = np.zeros((n_drugs, n_drugs))

    for i in range(n_drugs):
        for j in range(i, n_drugs):
            sim = compute_drug_similarity(drugbank_filtered['smiles'].iloc[i], drugbank_filtered['smiles'].iloc[j], morgan_gen)
            D_D_1[i, j] = sim
            D_D_1[j, i] = sim

    #D_D_2
    drug_side_effects = pharmacologically_active.groupby('Drug IDs')['Name'].apply(list).to_dict()
    side_effects = list(set(effect for effects in drug_side_effects.values() for effect in effects))

    side_effect_matrix = np.zeros((n_drugs, len(side_effects)))
    for i, drug in enumerate(drugbank_filtered['Drug id']):
        for effect in drug_side_effects.get(drug, []):
            side_effect_matrix[i, side_effects.index(effect)] = 1

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(side_effect_matrix)

    D_D_2 = np.zeros((n_drugs, n_drugs))
    for i in range(n_drugs):
        for j in range(n_drugs):
            D_D_2[i, j] = 1 if clusters[i] == clusters[j] and clusters[i] != -1 else 0

    #D_D_3
    drug_diseases = pharmacologically_active.groupby('Drug IDs')['Name'].apply(list).to_dict()

    D_D_3 = np.zeros((n_drugs, n_drugs))
    for i, drug1 in enumerate(drugbank_filtered['Drug id']):
        for j, drug2 in enumerate(drugbank_filtered['Drug id']):
            diseases1 = set(drug_diseases.get(drug1, []))
            diseases2 = set(drug_diseases.get(drug2, []))
            if diseases1 and diseases2:
                D_D_3[i, j] = len(diseases1.intersection(diseases2)) / len(diseases1.union(diseases2))

    # T-T-1: Target Interaction Relationships (based on sequence similarity)
    n_targets = len(proteins)
    T_T_1 = np.zeros((n_targets, n_targets))

    for i in range(n_targets):
        for j in range(i, n_targets):
            seq1 = Seq(proteins['sequence'].iloc[i])
            seq2 = Seq(proteins['sequence'].iloc[j])
            alignment_score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
            similarity = alignment_score / max(len(seq1), len(seq2))
            T_T_1[i, j] = T_T_1[j, i] = similarity

    # T-T-2: Target-Disease Relationships (based on sequence clustering)
    # We'll use DBSCAN clustering on the similarity matrix

    # Normalize similarity scores
    similarity_matrix = 1 - (T_T_1 / T_T_1.max())

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
    clusters = dbscan.fit_predict(similarity_matrix)

    T_T_2 = np.zeros((n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            T_T_2[i, j] = 1 if clusters[i] == clusters[j] and clusters[i] != -1 else 0

    D_D_1 = pd.DataFrame(D_D_1)
    D_D_2 = pd.DataFrame(D_D_2)
    D_D_3 = pd.DataFrame(D_D_3)
    T_T_1 = pd.DataFrame(T_T_1)
    T_T_2 = pd.DataFrame(T_T_2)

    D_D_1.to_csv("../../Datasets/processed/D_D_1.csv", index=False)
    D_D_2.to_csv("../../Datasets/processed/D_D_2.csv", index=False)
    D_D_3.to_csv("../../Datasets/processed/D_D_3.csv", index=False)
    T_T_1.to_csv("../../Datasets/processed/T_T_1.csv", index=False)
    T_T_2.to_csv("../../Datasets/processed/T_T_2.csv", index=False)

def create_dti_layer(AD1, AD2, AD3, AT1, AT2, AY):

    n_drugs = AD1.shape[0]
    n_targets = AT1.shape[0]

    I_drugs = np.eye(n_drugs)
    I_targets = np.eye(n_targets)

    # print(AY.shape)

    AM = np.block([
    [AD1,      AY     ],
    [AY.T,     AT2    ]])
    
    AM = pd.DataFrame(AM)
    AM.to_csv("../../Datasets/processed/AM.csv")

def main():
    pharma, target_labels, dti, drugbank, proteins, conf = load_data()
    targets = proteins['pdb_id'].tolist()

    # construct_similarity_matrices(drugbank, pharma, proteins)

    AD1 = pd.read_csv("../../Datasets/processed/D_D_1.csv").to_numpy()
    AD2 = pd.read_csv("../../Datasets/processed/D_D_2.csv").to_numpy()
    AD3 = pd.read_csv("../../Datasets/processed/D_D_3.csv").to_numpy()

    AT1 = pd.read_csv("../../Datasets/processed/T_T_1.csv").to_numpy()
    AT2 = pd.read_csv("../../Datasets/processed/T_T_2.csv").to_numpy()

    AY = pd.read_csv("../../Datasets/raw/target_labels.csv").filter(items = targets).to_numpy()

    create_dti_layer(AD1, AD2, AD3, AT1, AT2, AY)

if __name__ == "__main__":
    main()

    