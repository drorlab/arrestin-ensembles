import numpy as np
import os
from pensa import *
import argparse
import random

PDB = 'pdb/tremd-arr2-active_joint.pdb'
PDB_test = 'pdb/active_pdb.pdb'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=str)
    parser.add_argument('test', type=str)
    #parser.add_argument('outfile', help='Optional argument. Outfile name.', type=str, default='plots/clusters')
    args = parser.parse_args()
    reference = args.reference
    test = args.test
    #outfile = args.outfile
    ref_features, ref_data = get_structure_features(PDB, reference)
    test_features, test_data = get_structure_features(PDB_test, test)

    ref_data = ref_data["bb-torsions"]
    test_data = test_data["bb-torsions"]

    ref_features = ref_features["bb-torsions"]
    test_features = test_features["bb-torsions"]

    print(ref_features[:10])
    print(test_features[:10])

    ref_features, test_features, ref_data, test_data = get_common_features_data(ref_features, test_features, ref_data, test_data)

    ref_features, ref_data = sort_torsions_by_resnum(ref_features, ref_data)
    test_features, test_data = sort_torsions_by_resnum(test_features, test_data)

    print(ref_features[:10])
    print(test_features[:10])

    #print([i for i in ref_features + test_features if i not in ref_features or i not in test_features])
    test_metrics = {}
    
    test_metrics["avg_jsd"] = average_jsd(test_features, ref_features, test_data, ref_data)
    test_metrics["max_jsd"] = max_jsd(test_features, ref_features, test_data, ref_data)
    
    test_metrics["avg_kld"] = average_kld(test_features, ref_features, test_data, ref_data)
    test_metrics["max_kld"] = max_kld(test_features, ref_features, test_data, ref_data)
    
    
    test_metrics["avg_kss"] = average_kss(test_features, ref_features, test_data, ref_data)
    test_metrics["max_kss"] = max_kss(test_features, ref_features, test_data, ref_data)

    test_metrics["avg_ksp"] = average_ksp(test_features, ref_features, test_data, ref_data)
    test_metrics["max_ksp"] = max_ksp(test_features, ref_features, test_data, ref_data)
    test_metrics["min_ksp"] = min_ksp(test_features, ref_features, test_data, ref_data)

    test_metrics["pca_se"] = pca_sampling_efficiency(ref_data, test_data, num_pc = 4)
    
    #test_metrics["avg_ssi"] = average_ssi(ssi_features, ssi_features, bg, md, torsions='bb')
    #test_metrics["max_ssi"] = max_ssi(features, features, bg, md)

    print(f'test_metrics: {test_metrics}')

    with open('metrics.txt', 'w') as f:
        for metric, value in test_metrics.items():
            f.write(f'{metric} : {value}\n')


    

if __name__ == '__main__':
    main()
