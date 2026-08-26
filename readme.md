# NQS-DQME

Repository for these three papers:

[_NQS_](https://arxiv.org/abs/2404.11093)

[_PINN_](https://arxiv.org/abs/2404.11093)

[_Non-Markovian Error_](https://arxiv.org/abs/2608.22404)

## Contents

This repository is comprised of the following parts:

- data: the data in three papers above, the results for the third paper are not completely compatible with now library, please wait for a whole update of this library.

- examples: several examples with necessary data/information from DQME method such as the decomposition in the 'res_corr.data', truncated states in 'table.data', chem_potentials in the 'chem_potentials.data'. The 'input' file defines the structure of the RBM and some parameters in the TDVP method, while The 'input_h' file contains the hamiltonian information of the system.

- src/nqsdqme: the code of this library.

- tests: some test codes of this library.

- tutorials: several detail examples showing the structure and function of this repository.

## Installation

### Step 1: Install the required libraries

torch>=2.6.0+cu124

numpy>=2.2.6

scipy>=1.15.3

matplotlib>=3.10.7

When you install the above four libaries, all required libraries will be installed by pip.

Or you can install all libraries via `pip install -r requirement.txt`

### Step 2: Clone this repository

`git clone https://github.com/caolong-cn/NQS-DQME_fermions.git`

## Supported platforms

- CPU
- Nvidia GPU
