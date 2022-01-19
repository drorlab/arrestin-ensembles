rm slurm-* *output.pdb *output.dcd *.chk *_log.txt

for SB in submit_simulation*.sb; do
	sbatch $SB
done
