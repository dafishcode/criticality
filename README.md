# criticality
Repo contains code for algorithms which calculate the spatiotemporal propagation of activity (avalanches) and measures critical statistics.

## What is this repo for?
* the processing of avalanche dynamics
* the calculation of criticality statistics

## What does this repo contain?
* Modules contain functions which for avalanche and criticality analyses
* Accompanying ipynotebooks demonstrate how to use the modules

### Modules
'admin_functions.py' - useful administrative functions useful 

'IS.py' - module for Bayesian importance sampling approach for loglikelihood ratio testing of power law vs lognormal distribution

'criticality.py' - module for the estimation of avalanches and calculation of critical statistics

'trace_analyse.py' - classes for running multiple analyses on datasets


### Notebooks

'av_dev.ipynb' - criticality over development

'criticality_stats.ipynb' - critical statistics in spontaneous image data and null models

'criticality_seizure.ipynb' - criticality during PTZ induced seizure
