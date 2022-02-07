import MDAnalysis as md
import argparse
from pensa import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('top')
    parser.add_argument('traj')
    args = parser.parse_args()

    top = args.top
    traj = args.traj

    u = md.Universe(top, traj)
    ca = u.select_atoms("name CA")
    prev = None
    bond_list = []
    for c in ca.indices:
        if prev:
            bond_list.append((prev, c))
        prev = c
    u.add_bonds(bond_list)

    u_ca = md.Merge(ca)

    outfile = traj[:-4] + "_CA_bonds" + traj[-4:]
    if traj[-4:] == ".pdb":
        u_ca.atoms.write(outfile)
    else:
        with md.Writer(outfile, ca.n_atoms) as W:
            for ts in u.trajectory:
                W.write(ca.atoms)
            
            





if __name__ == "__main__":
    main()
