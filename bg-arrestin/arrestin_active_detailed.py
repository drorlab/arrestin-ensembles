#!/usr/bin/env python
# coding: utf-8

# # Alanine Dipeptide Sampling
# 
# ## Introduction
# 
# In this notebook, we will train a normalizing flow to generate conformations of a small molecule, alanine dipeptide.
# 
# We first import our packages
# - [bgflow](https://github.com/noegroup/bgflow/tree/factory) (for setting up the generator) 
# - [bmol](https://github.com/noegroup/bgmol) (for setting up the molecular system)
# 
# Note that some of the functionality used in this notebook is currently only implemented in the `factory` branch of bgflow. So `git checkout` that branch before running this notebook.
# 
# Please check out the list of requirements for these packages. For evaluating molecular potential energies of the conformations, you will have to set up a conda environment with [OpenMM](https://anaconda.org/conda-forge/openmm) and [openmmtools](https://anaconda.org/conda-forge/openmmtools) installed.

# In[7]:


import numpy as np
import torch
import bgflow as bg
import bgmol
import sys
import pickle
from bgmol.zmatrix import ZMatrixFactory

# In[8]:


#device = torch.device("cuda:0")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'torch.cuda.is_available: {torch.cuda.is_available()}')
dtype = torch.float32
to_tensor = lambda x: torch.tensor(x, dtype=dtype)
to_longtensor = lambda x: torch.tensor(x, dtype=torch.long)
to_array = lambda x: x.detach().cpu().numpy()


# ## System and Data
# 
# We first load a dataset that contains samples from a molecular dynamics (MD) simulation of the alanine dipeptide system.

# In[9]:


from bgmol.datasets import ArrestinActive

PDB = 'openmm-arrestin/arr2-active_output.pdb'
dataset = ArrestinActive()
dataset.read()
# The dataset contains MD samples and defines the target potential energy function that we are trying to sample from. We can access them as follows.

# In[10]:

print("Data shape:", dataset.coordinates.shape)
print('Trajectory type', type(dataset._trajectory))
#print('Trajectory shape', dataset._trajectory.shape)
print('Energies type', type(dataset._energies))
#print('Energies shape', dataset._energies.shape)
print('Forces type', type(dataset.forces))
#print('Forces shape', dataset._forces.shape)


N_WORKERS = 16  # set this to 1 to get best reliability
target = dataset.get_energy_model(n_workers=N_WORKERS)
print(type(target))

# The samples $r$ are atom coordinates the 22 atoms of alanine dipeptide `(n_samples x n_atoms x 3 spatial coordinates)`. The energy model is a function $u(r):\mathbb{R}^{66}\to\mathbb{R}$ and defines the target distribution up to a normalizing constant as $p(r) \sim e^{-u(r)}$. We can evaluate it as follows:

# In[11]:


data = to_tensor(dataset.coordinates.reshape(-1, 17100))
print(data.shape)
target.energy(data[:10])


# Using the packages [mdtraj](https://anaconda.org/conda-forge/mdtraj) and [nglview](https://anaconda.org/conda-forge/nglview), we can visualize conformations as follows.

energy_logfile = 'openmm-arrestin/arr2-active_log.txt'
target_energy = np.loadtxt(energy_logfile, delimiter=',')[:,1]


#is this necessary? because I think it calculates it from coordinates
#dataset.dim = np.prod(self.xyz[0].shape)

# In[12]:


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


# The most important feature of the Boltzmann distribution is the nontrivial joint marginal of the two central "backbone torsion angles" $\phi$ and $\psi$. We can visualize the distribution in a so called "Ramachandran" plot (basically just a 2D histogram). 

# In[13]:


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_torsions(ax, samples, filename='torsions.txt'):
    phi, psi = bgmol.systems.ala2.compute_phi_psi(to_mdtraj(to_array(samples)))
    with open('torsions/' + filename, 'w') as f:
        f.truncate(0)
        f.write('phi\n')
        for angle in phi:
            f.write(str(angle) + '\n')
        f.write('\n')
        f.write('psi\n')
        for angle in psi:
            f.write(str(angle) + '\n')
    print("Wrote phi_psi")
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
fig, ax = plt.subplots(figsize=(3,3))
plot_torsions(ax, data, 'ala2_md_phi_psi.txt')
plt.savefig('plots/ala2_detailed_phi_psi', bbox_inches='tight')


# ## (Optional:) Alchemically Modified System
# 
# One problem with sampling from this potential energy is that it contains singularities. Energies become infinite if any two atoms come very close. To ease the sampling task, we can replace the pairwise potential terms by "soft-core" potentials. This can be done using the alchemical factory from openmmtools. This step is not explained in detail. 

# In[14]:
'''
print(type(dataset.system.mdtraj_topology))

from openmmtools import alchemy
from simtk import openmm as mm

n_atoms = dataset.system.mdtraj_topology.n_atoms
region = alchemy.AlchemicalRegion(
    alchemical_atoms=np.arange(n_atoms), 
    annihilate_electrostatics=True, 
    annihilate_sterics=True,
    softcore_alpha=0.1,
    softcore_beta=0.1
)
factory = alchemy.AbsoluteAlchemicalFactory()
alchemical_system = factory.create_alchemical_system(dataset.system.system, region)
alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
alchemical_state.lambda_electrostatics = 0.99
alchemical_state.lambda_sterics = 0.99
alchemical_state.apply_to_system(alchemical_system)

alchemical_bridge = bg.OpenMMBridge(
    alchemical_system, 
    mm.LangevinIntegrator(dataset.temperature, 1, 0.001),
    n_workers=N_WORKERS
)
alchemical_target = bg.OpenMMEnergy(3*n_atoms, alchemical_bridge)
'''

# Now we have an approximate target energy that is never infinite and whose distribution has reasonable overlap with the target distribution.

# ## Coordinate Transform

# Next, we set up the normalizing flow that will be trained to sample from either $p(x)$ or the modified target.
# 
# To ease the sampling task, we will not directly sample Cartesian positions. Instead, we will sample internal coordinates (ICs), i.e., bond lengths, angles, and torsions between connected atoms. In this IC space, sampling becomes much easier and we automatically incorporate translation and rotation invariance of the potential. *(That said, if you want a much more difficult sampling task, you can try dropping the IC Layer and try training a flow from a 66-dimensional prior to the Cartesian coordinates directly).*
# 
# The transformation between internal coordinates and Cartesian coordinates will be the final (non-trainable) transform in the flow. We set it up as follows.

# In[9]:
#import MDAnalysis as md
#u = md.Universe(PDB)
#fa = u.select_atoms('name CA').indices
#fixed_atoms = to_longtensor(fa)

top = dataset.system.mdtraj_topology
factory = ZMatrixFactory(top, cartesian=top.select('name CA or (resname ACE and name C CH3 H1) or (resname NME and name C N H1)'))
zmatrix, fixed = factory.build_with_templates('z_protein.yaml', 'termini.yaml')

#print('zmatrix shape', zmatrix.shape)
#print('zmatrix', zmatrix)

coordinate_transform = bg.MixedCoordinateTransformation(
    z_matrix=zmatrix,
    normalize_angles=True,
    data=data,
    fixed_atoms=fixed
)


# The Z-matrix defines, which bonds, angles, and torsions should be used as internal coordinates. Let's inspect its output for our MD samples.

# In[10]:

#x0, rot, not returned
bonds, angles, torsions, z_fixed, dlogp = coordinate_transform.forward(data[:10])
print(bonds.shape, angles.shape, torsions.shape, z_fixed.shape, dlogp.shape)


# As for all bijective flow transforms in bgflow, the last return value is the log determinant of the Jacobian that we will use later to evaluate flow energies. The other values correspond to flow outputs, (here: bonds, angles, torsions and the system's origin and rotation). The rotation is given in Euler angles. All angles and torsions are normalized to $[0,1].$

# ## Normalizing Flow
# 
# We will ignore the global origin and orientation for now and just set it to a constant.
# 
# Still, setting up the normalizing flow takes a lot of care, as bonds, angles, and torsions live on different topologies. Bonds are in $(0,\infty),$ normalized angles in $[0, 1].$ Torsions live on the unit circle, so we need to make sure that the distribution is periodic, i.e. the normalized torsions are in $\mathbb{R}/[0,1].$
# 
# Bgflow has factory functions and a `BoltzmannGeneratorBuilder` class that takes care of this under the hood. We will now use this high-level API to set up the flow. 
# 
# The flow will use uniform prior distributions for bonds, angles, and torsions. We can parse the corresponding prior dimensions from the coordinate transform.

# In[11]:


shape_info = bg.ShapeDictionary.from_coordinate_transform(coordinate_transform)


# In[12]:


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


# The builder class uses reasonable default choices for the transforms and conditioner networks but it's customizable. If you want to tinker with the settings, take a look at the documentation.
# 
# For example each `add_condition` call can take various keyword arguments that specify the depth and width of the conditioner network, the transformer type, etc. To change the dimension and number of hidden layers in a conditioner, try something like `builder.add_condition(ANGLES, on=TORSIONS, hidden=(32,64,32))`.
# 
# By default, we use coupling layers with neural spline transforms (and circular spline transforms for torsions).

# In[13]:


from bgflow import TORSIONS, FIXED, BONDS, ANGLES

print("Torsions, fixed, bonds, angles: ")
for value in [TORSIONS, FIXED, BONDS, ANGLES]:
    #print("shape:", value.shape)
    print("type:", type(value))
    print("value:", value)
# In[14]:


builder.clear()

# split torsions in two channels
n_torsions = builder.current_dims[TORSIONS][-1]
n_t1 = n_torsions // 2
n_t2 = n_torsions - n_t1
TORSIONS_1, TORSIONS_2 = builder.add_split(
    TORSIONS, ["TORSIONS_1", "TORSIONS_2"], [n_t1, n_t2]
)
# condition the channels on each other
for i in range(4):
    builder.add_condition(TORSIONS_1, on=TORSIONS_2)
    builder.add_condition(TORSIONS_2, on=TORSIONS_1)
    builder.add_condition(FIXED, on=(TORSIONS_1, TORSIONS_2))
# concatenate torsions again and use them to inform bonds and angles
builder.add_merge([TORSIONS_1, TORSIONS_2], to=TORSIONS)
for i in range(4):
    builder.add_condition(ANGLES, on=(BONDS, TORSIONS))
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS))
builder.add_condition(ANGLES, on=(TORSIONS, FIXED))
builder.add_condition(BONDS, on=(ANGLES, TORSIONS, FIXED))
# all transforms so far were on [0,1]. We map the ICs to their domains and then apply
# the coordinate transform
marginal_estimate = bg.InternalCoordinateMarginals(
    builder.current_dims, builder.ctx,
    bond_lower=0.03, bond_upper=0.5, bond_mu=0.1, bond_sigma=1.0,
    angle_lower=0.1, angle_upper=.9, angle_mu=0.6, angle_sigma=1.0,
)
builder.add_map_to_ic_domains(marginal_estimate)
builder.add_map_to_cartesian(coordinate_transform, fixed_origin_and_rotation=True)
generator = builder.build_generator()


# In[15]:

print("data[:1]:", data[:1])
print("shape: ", data[:1].shape)
print(torch.sum(torch.isnan(data[:1])))
generator._flow.forward(data[:1].to(device), inverse=True)


# ## Train
# 
# Set up data loaders and optimizer.

# In[16]:


from torch.utils.data import TensorDataset, DataLoader

data_fraction = 0.8
n_data = int(data_fraction*len(data))

trainloader = DataLoader(
    TensorDataset(data[:n_data]),
    shuffle=True, batch_size=256
)


# In[17]:


optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)


# ### Training with NLL

# In[18]:


from tqdm.notebook import tqdm


# 
# 
# The normalizing flow can be trained by minimizing the negative log likelihood with respect to the data.

# In[19]:


n_epochs = 1

for epoch in tqdm(range(n_epochs)):
    # --- data based training ---
    for batch, in tqdm(trainloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = generator.energy(batch).mean()
        loss.backward()
        optimizer.step()
        print(f"{loss.item():.2f}", end="\r")


# ### Plot

# In[20]:


def ess(samples, inverse=False):
    """effective sample size"""
    du = generator.energy(samples)[...,0] - target.energy(samples)[...,0]
    if inverse:
        du = -du
    w = torch.nn.functional.softmax(-du, dim=0)
    return (w.sum())**2/(w**2).sum()


# In[28]:


def evaluate(generator=generator):
    generator.eval()
    samples = generator.sample(5000)

    fig, ax = plt.subplots(figsize=(3,3))
    #plot_torsions(ax, samples, 'ala2_bg_phi_psi.txt')
    #plt.savefig('ala2_detailed_bg_phis_psi', bbox_inches='tight')

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

    widget = show_trajectory(samples)
    del samples
    generator.train()
    return widget

evaluate()


# ### Training with Mixture of NLL and Reverse KLD

# In[29]:


constrain = lambda x: 1e4 - torch.nn.functional.elu(1e4 - x)


# In[32]:


n_epochs = 1
kld_weight = 0.2
nll_weight = 1.0 - kld_weight

for epoch in tqdm(range(n_epochs)):
    # --- data based training ---
    for batch, in tqdm(trainloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        nll = generator.energy(batch).mean()
        kld = constrain(generator.kldiv(trainloader.batch_size)).mean()
        loss = nll_weight * nll + kld_weight * kld
        loss.backward()
        optimizer.step()
        print(loss.item(), end="\r")


# In[33]:


evaluate()


# # Save and Load

# In[ ]:


#torch.save(generator.state_dict(), "saved_model.pt")


# In[ ]:


#generator.load_state_dict(torch.load("saved_model.pt"))
sys.exit()
