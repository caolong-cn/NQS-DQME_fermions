This tutorial is about how to utilize the this repository to simulate the evolution governed by DQME via PINN. It interprets the 'main_evolution.py' in the 'examples/pinn' floder. You can run `python main_evolution.py bfgs` in that floder.

This example approximates the evolution in $t\in[4.5,6]$ by a MLP.

## Step 1: Load the necessary modules

```python
import torch
import numpy as np
import sys
import time

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import scipy.optimize as spop
import copy

from nqsdqme.state.NADOt import MLPt
from nqsdqme.core.liouville import LiouvilletSaveL,Liouvillet
from nqsdqme.core.read import ReadHamilton
from nqsdqme.core.operators import Operators

from nqsdqme.core.basis import Basis
from nqsdqme.core.sampler import Sampler

from nqsdqme.utils import get_last_line,SubscriptTrans0_torch
from nqsdqme.global_defs import update_device,get_device

from nqsdqme.solver import PINNSolverSPOP,PINNSolverAdam
```

## Step 2: Load the hyperparameters in the 'input' file

```python
#load device
input_device = np.loadtxt("input",max_rows=1,dtype=np.str_)
device = torch.device(str(input_device))
update_device(device)
print(f'device:{device}')

#load MLP hyperparameters and define the problem to be solved 
input_para = np.loadtxt("input",skiprows=1,max_rows=3,dtype=np.int32)
input_t = np.loadtxt("input",skiprows=4,dtype=np.float64)
nhidden = input_para[0,0]
nonmccut = input_para[1,0]
allcut = input_para[1,1]
```

## Step 3: Initialize the liouville, RBM, basis, sampler and operators

```python
#Liouville: the class implementing the DQME and calculating <s|L|\rho(t)>
method = sys.argv[1]
if method=='bfgs':
    liouville = LiouvilletSaveL(ReadHamilton(),get_device(),nonmccut,allcut)
elif method=='adam':
    liouville = Liouvillet(ReadHamilton(),get_device(),nonmccut,allcut)

#initialize the nn
nstate = 2*liouville.nvar*liouville.nspin + liouville.nsgn*liouville.nvar*liouville.nspin*liouville.ncor*liouville.nalf
Ns = liouville.nvar * liouville.nspin
Nd = liouville.nsgn * Ns * liouville.nalf * liouville.ncor
statescols = [i for i in range(3,3+nstate)]
filename="table_cut"+str(nonmccut+1)+".data"
table = np.loadtxt("table_cut3.data",usecols=(statescols),dtype=np.int8)
#table0 consists of the non-zero states of rdo
condition = np.sum(table[:,:Nd],axis=1)==0
table0 = table[condition][:,Nd:]

rho = MLPt(liouville.nvar,liouville.nspin,liouville.ncor,liouville.nalf
           ,liouville.nsgn,get_device(),
           table0,N_t=1,gamma=-0.,nhidden=nhidden).to(get_device())


#initialize the basis and sampler
#Basis: the class saving the low order states
basis = Basis(liouville,nonmccut,allcut)

#Sampler: the class saving the high order samples
sampler = Sampler(basis,0,
                 nonmccut,allcut,
                 lmbda=-3.)


print('states_exact',':',basis.states.shape)
print(f'nhidden:{rho.nhidden:d}')
print(f'number of parameters:{rho.nparameters:d}',flush=True)
print(f'gamma:{rho.gamma}')


if method=='bfgs':
    liouville.set_states_need(rho,basis.states)

#initialize the operators
#Operators: the class computing the relating physical quantities 
#such as occupation numbers and electric currents
states_torch = torch.tensor(basis.states,device=get_device())
operators = Operators(liouville,states_torch)
coef = 3**torch.sum(states_torch[:,:rho.Nd],dim=1,dtype=torch.float64)
```

## Step 4: Initialize the TDVP solver and optimize the loss function

```python
w_b = 0.8
w_e = 1.-w_b
w_r = 20.
delta_t = 1e-8

t_0 = 4.5
t_end = 6.0
t_step = 0.2
residual_ts = torch.arange(t_0,t_end,t_step,dtype=torch.float64,device=get_device())
print(f'ts:{residual_ts}')


#loading initial conditions and initial rho

rho_name = 'rho_t_4.5_0.1'
print(rho_name)
rho.load_state_dict(torch.load(rho_name,map_location=get_device())) 
print(f't_0:{t_0}')
if t_0<1.e-6:
    rho_input = torch.from_numpy(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=(0,1)))
    target = (rho_input[:,0] + 1.j*rho_input[:,1]).to(get_device())
else:
    rho1 = copy.deepcopy(rho)
    rho1.load_state_dict(torch.load('rho_t_4.5_0.1',map_location=device))
    target = rho1.States(states_torch,residual_ts[0]).detach()

if method == 'bfgs':
    solver = PINNSolverSPOP(liouville,
                    operators,
                    basis,residual_ts,rho,
                    coef,target,
                    w_b ,w_e,w_r,delta_t
                    )
    vec_of_rho = rho.nn_to_vec().to("cpu").numpy()
    # print(vec_of_rho)
    print(f'loss:{solver.loss(vec_of_rho)}')
    result = solver.solve()
    solver.print_result(result)

    torch.save(rho.state_dict(),'rho_BFGS_ll_t_'+str(rho.nhidden))
    rho_0 = rho.rho_0().detach()
    print(f'origin_trace: {torch.real(torch.trace(rho_0)): .3e}\n',flush=True)
    print(f'rho_0: {rho_0}')
elif method=='adam':
    #This method is not completed.
    solver = PINNSolverAdam(liouville,
                    operators,
                    basis,residual_ts,rho,
                    coef,target,
                    w_b ,w_e,w_r,delta_t
                    )
    print(f'loss:{solver.loss(solver.states,solver.target_initial,solver.coef_initial)}')
    print(f'loss:{solver.loss(solver.states,solver.target_initial,solver.coef_initial)}')

    solver.solve(1000)
```
