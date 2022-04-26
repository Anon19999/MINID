# Mixed Integer Neural Inverse Design

This repository contains code for the following paper:

+ [Mixed Integer Neural Inverse Design (arxiv)](https://arxiv.org/pdf/2109.12888.pdf)

## Requirements
YALMIP Version 31-March-2021

Gurobi v 9.01


## Instructions
For the matlab code, install `Gurobi` and `Yalmip`, and add yalmip to the matlab path.

Download and extract the data from [Here](https://drive.google.com/file/d/1TQIqNVP4qo3L9VL7IBjtCDmYwHqxViAg/view?usp=sharing)

Run `main.m` and follow the given instuctions to run each experiment or plot the results.


## Code Structure

The code for each experiment is in a separate directory. Each directory corresponds to the experiments as follows:

+ `Spectral Separation`: Neural Spectral Separation from section 4.1
+ `Softrobot`: Robot Inverse Kinematic from section 4.2
+ `Inversion_and_selection`: Material Selection from section 4.3
+ `Nano-Photonics`: Nano-Photonics from section 4.4.1
+ `Contoning`: Contoning from section 4.4.2
+ `MILP_NA_combination`: Combination of MILP and NA from section 4.5
+ `Robustness`: Robustness Analysis from section 4.6
