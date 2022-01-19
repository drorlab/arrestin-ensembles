import mdtraj as md
from nma import ANMA
import os
import argparse

DIRNAME = "./pdb_final"
DIRNAME = "./pdb_minim"
DIRNAME = "./ala2"
DIRNAME = "./pdb_final/tremd"
NUM_MODES = 3

def compute_nma(filename, rmsd):
    pdb = md.load(f'{DIRNAME}/{filename}')
    path = f'./trajectories/{filename[:-4]}_{rmsd * 10}'
    if not os.path.isdir(path):
        os.mkdir(path)
    print(f'Computing NMA for {filename} ...')
    anmas = [ANMA(mode=i, rmsd=rmsd, n_steps=5000, selection='all') for i in range(NUM_MODES)]
    trajs = [anma.fit_transform(pdb) for anma in anmas]
    for i in range(len(trajs)):
        trajs[i].save(f'{path}/{filename[:-4]}_{rmsd * 10}_{i}.xtc')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rmsd', help='Compute NMA on pdb files using provided rmsd',
            type=float)
    args = parser.parse_args()
    rmsd = args.rmsd / 10
    for filename in os.listdir(DIRNAME):
        if filename.endswith('.pdb'):
            compute_nma(filename, rmsd)

if __name__ == '__main__':
    main()
