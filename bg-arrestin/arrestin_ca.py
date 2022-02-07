#!/usr/bin/env python
# coding: utf-8


# In[1]:


import numpy as np
import torch
import bgflow as bg
import bgmol
import sys
import pickle

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from bgmol.zmatrix import ZMatrixFactory


# In[2]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--num_epochs_nll', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--outfile', type=str, default="arrestin_active")
parser.add_argument('--statefile', default="")
parser.add_argument('--write_checkpoint', type=bool, default=False)
parser.add_argument('--read_checkpoint', type=bool, default=False)

args = parser.parse_args()

statefile = args.statefile

# In[3]:


torch.manual_seed(args.random_seed)


# In[4]:


#device = torch.device("cuda:0")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'torch.cuda.is_available: {torch.cuda.is_available()}')
dtype = torch.float32
to_tensor = lambda x: torch.tensor(x, dtype=dtype)
to_longtensor = lambda x: torch.tensor(x, dtype=torch.long)
to_array = lambda x: x.detach().cpu().numpy()


# ## System and Data
# 
# We first load a dataset that contains samples from a molecular dynamics (MD) simulation.

# In[5]:


from bgmol.datasets import ArrestinActiveCA

dataset = ArrestinActiveCA()
dataset.read()


# The dataset contains MD samples and defines the target potential energy function that we are trying to sample from. We can access them as follows.

# In[6]:


print("Data shape:", dataset.coordinates.shape)
print('Trajectory type', type(dataset._trajectory))
#print('Trajectory shape', dataset._trajectory.shape)
print('Energies type', type(dataset._energies))
#print('Energies shape', dataset._energies.shape)
print('Forces type', type(dataset.forces))
#print('Forces shape', dataset._forces.shape)

N_WORKERS = 1  # set this to 1 to get best reliability
target = dataset.get_energy_model(n_workers=N_WORKERS)
print(type(target))


# The samples $r$ are atom coordinates `(n_samples x n_atoms x 3 spatial coordinates)`. 
# The energy model is a function $u(r):\mathbb{R}^{66}\to\mathbb{R}$ and defines the target distribution up to a normalizing constant as $p(r) \sim e^{-u(r)}$.
# We can evaluate it as follows:

# In[7]:


data = to_tensor(dataset.coordinates.reshape(-1, 17100))
print(data.shape)
target.energy(data[:10])


# Using the packages [mdtraj](https://anaconda.org/conda-forge/mdtraj) and [nglview](https://anaconda.org/conda-forge/nglview), we can visualize conformations as follows.

# In[8]:


import mdtraj as md
import nglview as nv

def to_mdtraj(data, topology=dataset.system.mdtraj_topology):
    return md.Trajectory(
        data.reshape([-1, 5700, 3]),
        topology=dataset.system.mdtraj_topology
    )
    
def show_trajectory(data):
    widget = nv.show_mdtraj(to_mdtraj(to_array(data)))
    return widget

show_trajectory(data[:10])


# We can construct the internal representation using the Z matrix factory: 

# In[9]:


# Topology of the system
top = dataset.system.mdtraj_topology
# The following atoms will be treated with cartesian coordinates.
selection = 'name CA'
# Define the factory.
factory = ZMatrixFactory(top, cartesian=top.select(selection))
# Build the internal representation from templates defined in yaml files.
zmatrix, fixed = factory.build_with_templates('z_protein.yaml', 'termini.yaml')

state = {}
if statefile != "":
    with open(statefile, 'rb') as f:
        state = pickle.load(f)
        

# In[10]:


atomnames = [str(atom) for atom in top.atoms]
print(atomnames[2], end="\n\n")
print("; ".join(atomnames[408:419]), end="\n\n")
print("; ".join(atomnames[1748:1759]), end="\n\n")
print(atomnames[1977], end="\n\n")
print(atomnames[2180], end="\n\n")
print(atomnames[2330], end="\n\n")
print("; ".join(atomnames[2466:2477]), end="\n\n")
print("; ".join(atomnames[3119:3130]), end="\n\n")
print("; ".join(atomnames[3320:3331]), end="\n\n")
print("; ".join(atomnames[3450:3460]), end="\n\n")
print(atomnames[3842], end="\n\n")
print(atomnames[3990], end="\n\n")
print(atomnames[4232], end="\n\n")
print("; ".join(atomnames[4658:4669]), end="\n\n")
print("; ".join(atomnames[5551:5561]), end="\n\n")
print(atomnames[5695], end="\n\n")
print(atomnames[5698], end="\n\n")


# In[11]:


coordinate_transform = bg.MixedCoordinateTransformation(
    z_matrix=zmatrix,
    normalize_angles=True,
    data=data,
    fixed_atoms=fixed,
    keepdims=len(fixed) - 3
).to(device)

#x0, rot, not returned
bonds, angles, torsions, z_fixed, dlogp = coordinate_transform.forward(data[:10].to(device))
print(bonds.shape, angles.shape, torsions.shape, z_fixed.shape, dlogp.shape)


# In[12]:


shape_info = bg.ShapeDictionary.from_coordinate_transform(coordinate_transform)
print(shape_info)


# In[13]:


# you can choose here, if you want to use the target with singularities
# or the approximation without singularities
TARGET = target 
#TARGET = alchemical_target

builder = bg.BoltzmannGeneratorBuilder(
    shape_info, 
    target=TARGET,
    device=device, 
    dtype=dtype
)

print(builder.current_dims)


# The builder class uses reasonable default choices for the transforms and conditioner networks but it's customizable. If you want to tinker with the settings, take a look at the documentation.
# 
# For example each `add_condition` call can take various keyword arguments that specify the depth and width of the conditioner network, the transformer type, etc. To change the dimension and number of hidden layers in a conditioner, try something like `builder.add_condition(ANGLES, on=TORSIONS, hidden=(32,64,32))`.
# 
# By default, we use coupling layers with neural spline transforms (and circular spline transforms for torsions).

# In[14]:


from bgflow import TORSIONS, FIXED, BONDS, ANGLES


# In[15]:


builder.clear()

# Split torsions in two channels
n_torsions = builder.current_dims[TORSIONS][-1]
n_t1 = n_torsions // 2
n_t2 = n_torsions - n_t1
TORSIONS_1, TORSIONS_2 = builder.add_split(
    TORSIONS, ["TORSIONS_1", "TORSIONS_2"], [n_t1, n_t2]
)

# Condition the channels on each other
for i in range(4):
    builder.add_condition(TORSIONS_1, on=TORSIONS_2)
    builder.add_condition(TORSIONS_2, on=TORSIONS_1)
builder.add_merge([TORSIONS_1, TORSIONS_2], to=TORSIONS)

for i in range(4):
    builder.add_condition(TORSIONS, on=FIXED)
    builder.add_condition(FIXED, on=TORSIONS)
    
# Concatenate torsions again and use them to inform bonds and angles
for i in range(4):
    builder.add_condition(ANGLES, on=(BONDS, TORSIONS))
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS))

# All transforms so far were on [0,1].
# We map the ICs to their domains and then apply the coordinate transform.
marginal_estimate = bg.InternalCoordinateMarginals(
    builder.current_dims, builder.ctx,
    bond_lower=0.03, bond_upper=2.0, bond_mu=0.1, bond_sigma=1.0,
    angle_lower=0.05, angle_upper=1.0, angle_mu=0.6, angle_sigma=1.0,
)
builder.add_map_to_ic_domains(marginal_estimate)
#builder.add_map_to_cartesian(coordinate_transform, fixed_origin_and_rotation=True)

print(builder.current_dims)
generator = builder.build_generator()


# In[16]:


generator._flow


# In[17]:


print("data[:1]:", data[:1])
print("shape: ", data[:1].shape)

generator._flow.forward(data[:1].to(device), inverse=True)


# ## Training
# 
# Set up data loaders and optimizer.

# In[18]:


from torch.utils.data import TensorDataset, DataLoader

data_fraction = 0.8
n_data = int(data_fraction*len(data))

trainloader = DataLoader(
    TensorDataset(data[:n_data]),
    shuffle=True, batch_size=16
)

optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)


# In[19]:


def ess(samples, inverse=False):
    """effective sample size"""
    du = generator.energy(samples)[...,0] - target.energy(samples)[...,0]
    if inverse:
        du = -du
    w = torch.nn.functional.softmax(-du, dim=0)
    return (w.sum())**2/(w**2).sum()

def write_traj(samples, outfile='arrestin_active'):
    num_samples = samples.shape[0]
    xyz = samples.detach().cpu().numpy().reshape(num_samples, -1, 3)
    trj = md.Trajectory(
        xyz,
        topology=dataset.system.mdtraj_topology
    )
    trj.save_dcd(outfile + ".dcd")
    trj.save_pdb(outfile + ".pdb")

def evaluate(generator, out_filename):
    generator.eval()
    samples = generator.sample(args.num_samples)
    print("Samples Shape:", samples.shape)
    effective_sample_size = ess(samples)
    ess_inverse = ess(samples, inverse=True)
    print(f"Effective Sample Size: "
          f"{effective_sample_size.item():6.2f} "
          f"{ess_inverse.item():6.2f}"
    )
    print(f"Sampling Efficiency:  "
          f"{effective_sample_size / len(samples):6.6f}  "
          f"{ess_inverse.item() / len(samples):6.6f}"
    )
    write_traj(samples, out_filename)
    del samples
    

### Training with NLL

# In[20]:


from tqdm.notebook import tqdm


# The normalizing flow can be trained by minimizing the negative log likelihood with respect to the data.

# In[21]:
run_info = f"{args.num_samples}s_{args.num_epochs}e_{args.random_seed}seed_"

n_epochs_nll = args.num_epochs_nll
num_iters = 0
if args.read_checkpoint and statefile != "" and state.loop == 1:
    num_iters = state.epoch
    generator = state.generator
for epoch in tqdm(range(num_iters, n_epochs_nll)):
    # --- data based training ---
    if args.write_checkpoint and num_iters % 5 == 4:
        with open(f"generator_states/{run_info}{num_iters}epoch_loop1.pkl", "wb") as f:
            curr_state = {}
            curr_state["generator"] = generator
            curr_state["epoch"] = num_iters
            curr_state["loop"] = 1
            pickle.dump(curr_state, f)
    print(f"Num epochs completed: {num_iters}", flush=True)
    for batch, in tqdm(trainloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        try:
            loss = generator.energy(batch).mean()
            loss.backward()
        except:
            print('Loss calculation failed.')
        optimizer.step()
        print(f"{loss.item():.2f}", end="\r")
    num_iters += 1


# In[22]:

out = f"output/{args.outfile}_{args.num_samples}s_{args.num_epochs_nll}plus{args.num_epochs}e_data-based_sigma0.1_{args.random_seed}"
evaluate(generator, out)






