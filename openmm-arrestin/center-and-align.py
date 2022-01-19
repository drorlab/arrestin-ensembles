import numpy as np
import os
from pensa import *
import argparse
import random
import MDAnalysis as md


parser = argparse.ArgumentParser()
parser.add_argument('top_in', type=str)
parser.add_argument('trj_in', type=str)
args = parser.parse_args()


def center_structure(infile):
    u = md.Universe(infile)
    atoms = u.select_atoms('all')
    cm = atoms.center_of_mass()
    u.atoms.translate(-cm)
    outfile = infile[:-4] + '_centered.pdb'
    u.atoms.write(outfile)
    return outfile


def align_trajectory(ref, trj):
    outfile = trj[:-4] + '_aligned_final'
    align_coordinates(ref, ref, [trj], outfile)


if __name__ == '__main__':
    top_out = center_structure(args.top_in)
    align_trajectory(top_out, args.trj_in)


