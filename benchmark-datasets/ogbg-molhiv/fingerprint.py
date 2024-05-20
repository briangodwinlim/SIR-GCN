import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def generate_fingerprint(name, root, fp_types, save=True):
    smiles = pd.read_csv(os.path.join(root, name.replace('-', '_'), 'mapping/mol.csv.gz'))
    smiles = smiles['smiles'].tolist()
    fingerprints = {'morgan': [], 'maccs': [], 'rdkit': []}

    for i in tqdm(range(len(smiles))):
        rdkit_mol = Chem.MolFromSmiles(smiles[i])
        
        if fp_types.morgan:
            fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2)
            fingerprints['morgan'].append(fp)

        if fp_types.maccs:
            fp = MACCSkeys.GenMACCSKeys(rdkit_mol)
            fingerprints['maccs'].append(fp)

        if fp_types.rdkit:
            fp = Chem.RDKFingerprint(rdkit_mol)
            fingerprints['rdkit'].append(fp)
            
    for fp_type, fps in fingerprints.items():
        if len(fps) == 0:
            continue

        fingerprints[fp_type] = np.array(fps, dtype=np.int64)
        print(f'{fp_type} feature shape: {fingerprints[fp_type].shape}')

        if save:
            os.makedirs(os.path.join(root, name.replace('-', '_'), 'fingerprint'), exist_ok=True)
            np.save(os.path.join(root, name.replace('-', '_'), f'fingerprint/{fp_type}.npy'), fps)
            
    return fingerprints


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'Generate molecular fingerprint with RDKit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--dataset', type=str, default='ogbg-molhiv', help='name of dataset')
    argparser.add_argument('--root', type=str, default='dataset', help='root directory of dataset folder')

    argparser.add_argument('--morgan', action='store_true', help='generate Morgan fingerprint')
    argparser.add_argument('--maccs', action='store_true', help='generate MACCS keys')
    argparser.add_argument('--rdkit', action='store_true', help='generate RDKit topological fingerprint')

    argparser.add_argument('--save', action='store_true', help='save generated fingerprints')
    args = argparser.parse_args()
    
    generate_fingerprint(args.dataset, args.root, args, args.save)
