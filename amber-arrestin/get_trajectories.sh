DIR='/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories/incite-analysis/analysis_pensa/traj'
for CONDITION in active inactive v2rpp; do
	cp ${DIR}/arrestin2-${CONDITION}-tremd.??? .
done
