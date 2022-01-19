import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
import argparse
from sys import stdout
import pickle


# Parse user input and options
ap = argparse.ArgumentParser()
ap.add_argument('--condition', type=str, default='arr2-inactive',
                help='Folder name of the INCITE trajectory.')
ap.add_argument('--steps', type=int, default=1000000,
                help='Number of simulation steps.')
cmd = ap.parse_args()

condition = cmd.condition


# Define the force field
# AMBER99sb-ildn in combination with GBSA-OBC:
# https://pubmed.ncbi.nlm.nih.gov/15048829/.
forcefield = app.ForceField("amber99sbildn.xml", "amber96_obc.xml")

# Read topology and structure
directory = '/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories'
psf = directory+'/tremd-'+condition+'/system.psf'
crd = directory+'/tremd-'+condition+'/system.inpcrd'
top = app.CharmmPsfFile(psf).topology
pos = app.AmberInpcrdFile(crd).positions

# Construct and add box
edge_length = 20 # nm
lx = mm.Vec3(edge_length, 0, 0)
ly = mm.Vec3(0, edge_length, 0)
lz = mm.Vec3(0, 0, edge_length)
top.setPeriodicBoxVectors([lx,ly,lz])
box = top.getPeriodicBoxVectors()
print(box)

# Create the modeller object
modeller = app.Modeller(top, pos)

# Remove solvent (water + ions)
solvent = [res for res in modeller.topology.residues() if res.name in ['CLA','SOD','HOH']]
modeller.delete(solvent)

# Write start file
start_fname = condition+'_start.pdb'
app.PDBFile.writeFile(
    modeller.topology, modeller.positions, 
    open(start_fname, 'w'), keepIds=True
)

# Create the system from the start file
pdb = app.PDBFile(start_fname)
system = forcefield.createSystem(
    pdb.topology, 
    nonbondedMethod = app.PME, 
    nonbondedCutoff = 1*unit.nanometer, 
    constraints = app.HBonds
)
name = condition+"_system"
with open(name + '.pkl', 'wb') as f:
    pickle.dump(system, f, pickle.HIGHEST_PROTOCOL)

# Define the integrator
integrator = mm.LangevinIntegrator(
    310*unit.kelvin,
    1/unit.picosecond,
    0.002*unit.picoseconds
)

# Set up the simulation
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Run energy minimization
simulation.minimizeEnergy()

# Write minimized structure
minim_fname = condition+'_minim.pdb'
app.PDBFile.writeFile(
    modeller.topology, simulation.context.getState(getPositions=True).getPositions(),
    open(minim_fname, 'w'), keepIds=True
)

# Define the output
reporter_pdbtrj = app.PDBReporter(
    condition+'_output.pdb', 
    5000
)
reporter_dcdtrj = app.DCDReporter(
    condition+'_output.dcd',
    5000
)
reporter_stdout = app.StateDataReporter(
    stdout, 
    5000, 
    step=True, 
    potentialEnergy=True, 
    temperature=True
)
reporter_logtxt = app.StateDataReporter(
    open(condition+'_log.txt','w'),
    5000,
    step=True,
    potentialEnergy=True,
    temperature=True
)
reporter_checkp = app.CheckpointReporter(
    condition+'_checkpoint.chk', 
    5000
)
simulation.reporters.append(reporter_pdbtrj)
simulation.reporters.append(reporter_dcdtrj)
simulation.reporters.append(reporter_stdout)
simulation.reporters.append(reporter_logtxt)
simulation.reporters.append(reporter_checkp)

# simulation.loadCheckpoint(condition+'_checkpoint.chk')

# Run the simulation
simulation.step(cmd.steps)

