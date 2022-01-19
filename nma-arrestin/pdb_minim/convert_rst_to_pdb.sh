export AMBERHOME=/home/groups/rondror/software/amber16_gnu
source $AMBERHOME/amber.sh

DIR=/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories/incite-setup/remd-preparation

for SIM in arr2-active arr2-inactive arr2-v2rpp; do
    cpptraj -p $DIR/setup-$SIM/prep/system.prmtop -y $DIR/setup-$SIM/rep1/Min_3.rst -x ${SIM}_min.pdb
    cp $DIR/setup-$SIM/prep/system.prmtop ${SIM}.prmtop
    cp $DIR/setup-$SIM/prep/system.pdb ${SIM}.pdb
    cp $DIR/setup-$SIM/prep/system.psf ${SIM}.psf
done
