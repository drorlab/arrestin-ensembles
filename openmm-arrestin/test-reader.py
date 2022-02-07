import mdtraj
import MDAnalysis as mda

# Load data with mdtraj and print the first frame
data_mdt = mdtraj.load_dcd('arr2-active_output.dcd','arr2-active_start.pdb')
print('\nData as read by mdtraj:')
print(data_mdt.xyz[0])

# Load the same data with MDAnalysis and print the first frame
u = mda.Universe('arr2-active_start.pdb','arr2-active_output.dcd')
for ts in u.trajectory[:1]:
    pos = u.select_atoms('all').positions
print('\nData as read by MDAnalysis:')
print(pos)
print('')
