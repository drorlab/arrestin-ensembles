import os
from pensa import *
import argparse
import numpy as np
import random

PDB = 'pdb/tremd-arr2-inactive_joint.pdb'
NUM_PC = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', help="Reference trajectory", type=str)
    parser.add_argument('test', help="Test trajectory", type=str)

    args = parser.parse_args()
    reference = args.reference
    test = args.test

    _, test_arr = get_structure_features(PDB, test, features=['bb-torsions'])
    
    test_data = test_arr['bb-torsions']
    
    _, ref_arr = get_structure_features(PDB, reference, features=['bb-torsions'])

    ref_data = ref_arr['bb-torsions']

    pca_se = pca_sampling_efficiency(ref_data, test_data, num_pc=NUM_PC)

    print(pca_se)

if __name__ == "__main__":
    main()
