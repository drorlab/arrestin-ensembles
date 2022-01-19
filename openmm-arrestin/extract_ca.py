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

    outfile = traj[:-4] + "_CA" + traj[-4:]
    with md.Writer(outfile, ca.n_atoms) as W:
        for ts in u.trajectory:
            W.write(ca)





if __name__ == "__main__":
    main()
