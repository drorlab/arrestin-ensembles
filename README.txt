ala2 directory

Contains scripts, data, and plots related to sampling structural ensembles from alanine dipeptide.

Directories: 
              bg-ala2
                  contains plots produced from sampling from boltzmann generators
              md-ala2
                  contains plots produced from reference molecular dynamics ensembles
              nma-ala2
                  contains plots produced from sampling from normal mode analysis
              torsions
                  contains backbone torsion angles from bg and md simulations
              trajectories
                  contains normal modes produced from normal mode analysis
              plots 
                  contains plots that don't fit into an above directory


Files:
              alanine_dipeptide_basics.py
              alanine_dipeptide_detailed.py
              alanine_dipeptide_spline.py

                  scripts written by Andreas Kramer that run BG sampling
              
              alanine-dipeptide.pdb
                  
                  pdb file for alanine dipeptide

              get_clusters.py
              get_combined_clusters.py

                  scripts to plot clusters from torsional angles

              make_plots.py
              plot_data.py

                  scripts to create histograms and ramachandran plots from sampled trajectories.                   make_plots.py is more up-to-date.

              metrics_table.py

                  script that calculates and prints metrics evaluating sampling performance 
                  compared to reference ensembles.

              submit-training.sb
                
                  file used to submit jobs on sherlock

