# SPLICE

# README

This is the repository for the splicing method.

The splcing method is a methodology for predicting peptide molecular structures. 

# Required Packages

* python==3.8.0
* openmm==8.0.0
* mpi4py==3.1.4

Prothod_mquchong_tfd_initial_thresh.c and relax.static.linuxgccrelease (ROSETTA) should be present in $PATH

# USAGE

```shell
export OPENMM_CPU_THREADS=1
NP=8
ST='GeWTWddATkTWTWTe,nt=pro,ct=nh2(GeWT(Ge)(WT))(WddATkTW(WddA(Wd)(dA))(TkTW(Tk)(TW)))(TWTe(TW)(Te))'
mpiexec -np $NP python begin.py cpu -st $ST
```
