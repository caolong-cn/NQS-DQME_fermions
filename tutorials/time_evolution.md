This tutorial is about how to utilize the this repository to simulate the evolution governed by DQME via RBM ansatz. It interprets the 'main_evolution.py' in the 'examples' floder. You can run `python main_evolution.py` in that floder.

## Step 1: Load the necessary modules

```python
import torch
import numpy as np
import sys
import time
import os

#run this code in the 'examples' floder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from nqsdqme.state.NADO import RBM,RBM_nonds
from nqsdqme.core.liouville import LiouvilleSaveL
from nqsdqme.core.read import ReadHamilton
from nqsdqme.core.operators import Operators

from nqsdqme.core.basis import Basis
from nqsdqme.core.sampler import Sampler

from nqsdqme.solver import TDVP, TDVP_mc
from nqsdqme.solver import TDVP_test,TDVP_mc_test
from nqsdqme.utils import get_last_line
from nqsdqme.global_defs import update_device,get_device
```

## Step 2: Load the hyperparameters in the 'input' file

```python
#load device
input_device = np.loadtxt("input",max_rows=1,dtype=np.str_)
device = torch.device(str(input_device))
update_device(device)
print(f'device:{device}')


#load RBM hyperparameters and define the problem to be solved  
input_para = np.loadtxt("input",skiprows=1,max_rows=3,dtype=np.int32)
nhidden = input_para[0,0]
nauxillary = input_para[0,1]

#load TDVP hyperparameters
epsilonmode = input_para[0,2]
nonmccut = input_para[1,0]
allcut = input_para[1,1]
notsaving = input_para[1,2]
mc_size = input_para[2,0]
input_t = np.loadtxt("input",skiprows=4,max_rows=1,dtype=np.float64)
```

## Step 3: Initialize the liouville, RBM, basis, sampler and operators

```python
#Liouville: the class implementing the DQME and calculating <s|L|\rho>
liouville = LiouvilleSaveL(ReadHamilton(),get_device(),nonmccut,allcut,notsaving,mc_size=mc_size)
if nonmccut<allcut:
    liouville.lmbda = -3.

#initialize the RBM
nstate = 2*liouville.nvar*liouville.nspin + liouville.nsgn*liouville.nvar*liouville.nspin*liouville.ncor*liouville.nalf
Ns = liouville.nvar * liouville.nspin
Nd = liouville.nsgn * Ns * liouville.nalf * liouville.ncor
statescols = [i for i in range(3,3+nstate)]
filename="table_cut"+str(nonmccut+1)+".data"
table = np.loadtxt("table_cut3.data",usecols=(statescols),dtype=np.int8)
#table0 consists of the non-zero states of rdo
condition = np.sum(table[:,:Nd],axis=1)==0
table0 = table[condition][:,Nd:]
rho = RBM(liouville.nvar,liouville.nspin,liouville.ncor,liouville.nalf,liouville.nsgn,nhidden,nauxillary,get_device(),table0).to(get_device())
#load the initial RBM
#if you want to solve the steady states, you can remove the following 
#two statements
count = int(input_t[2])
rho.load_state_dict(torch.load('para'+str(count),map_location=get_device())) 

#initialize the basis and sampler
#Basis: the class saving the low order states
basis = Basis(liouville,nonmccut,allcut)
#Sampler: the class saving the high order samples
sampler = Sampler(basis,mc_size,
                 nonmccut,allcut,
                 lmbda=-3.)

print('states_exact',':',basis.states.shape)
print(f'nhidden:{rho.Nh:d}, nauxillary:{rho.Na:d}')
print(f'number of parameters:{rho.nparameters:d}',flush=True)

#initialize the Llocl computation in the liouville
if notsaving==0 :
    liouville.set_states_need(rho,basis.states)
    if nonmccut<allcut:
        liouville.set_states_need_mc(rho,sampler.samples)

        print(f'time of mc intialization:{t1-t0}')
        print(f'sampling number:{sampler.N_samples}')

#initialize the operators
#Operators: the class computing the relating physical quantities 
#such as occupation numbers and electric currents
states_torch = torch.tensor(basis.states,device=get_device())
operators = Operators(liouville,states_torch)
```

## Step 4: Initialize the TDVP solver

```python
input_epsilon = np.loadtxt("input",skiprows=5,dtype=np.float64)
t_lower = input_t[0]
t_upper = input_t[1] #200

if epsilonmode==0:
    emode = 'mid' 
elif epsilonmode==1:
    emode = 'quick'
elif epsilonmode==2:
    emode = 'slow'

if sampler.nonmccut<sampler.allcut:
    t0 = time.time()
    for i in range(100):
        sampler.flip()
    t1 = time.time()
    print(f'time of intial flip:{t1-t0}')
    # states_mc = torch.tensor(sampler.samples,device=get_device())
    tdvp = TDVP_mc(basis,sampler,rho,liouville,operators,
                    t_lower=t_lower,t_upper=t_upper,
                    atol=1e-5,k_S=10000,
                    epsilon=input_epsilon[0],min_epsilon=input_epsilon[1],mode=emode,
                    lmbda=liouville.lmbda,nonmccut=liouville.nonmccut,allcut=liouville.allcut)
else :
    tdvp = TDVP(basis,rho,liouville,operators,
                t_lower=t_lower,t_upper=t_upper,
                atol=1e-5,k_S=10000,
                epsilon=input_epsilon[0],min_epsilon=input_epsilon[1],mode=emode)
print(f'epsilon tuning mode: {tdvp.tuning_mode}')
```

## Step 5: Simulate the evolution

```python
tdvp.ischolesky = True #determine wheter chooses cholesy decompositiom
ifstop = False #determine whether still evolves when L2>1
while tdvp.iterator.t < t_upper :
    #perform one step evolution
    msg = tdvp.iterator.step()

    #save and print the related info
    count += 1
    print('here print the real step of t')
    t0 = time.time()
    rho_0 = rho.rho_0().detach()
    trace_rho0 = torch.real(torch.trace(rho_0))
    print(f'msg_rk45:{msg}')
    f1 = open('t-step','a')
    I = operators.current_general(rho,rho_0)
    Ib = torch.sum(I,dim=0)
    I_tot = torch.sum(Ib).reshape(1)
    print(f'current:{I}')
    f1.write(f'{get_last_line("t-n_all").decode()}')
    f1.write('      ')
    if rho.Nb == 1 : #rho.Nb=1: tt, j_left_u, j_left_d, j_left_u+j_left_d
        f1.write(f'{torch.real(I[0][0]).item(): .5e}  {torch.real(I[1][0]).item(): .5e}  {torch.real(I_tot[0]).item(): .5e}')
    else : #rho.Nb>1: tt, j_left, j_left, j_left+j_left
        I_tot = torch.cat((Ib,I_tot))
        for i in range(rho.Nb+1) : 
            f1.write(f'{torch.real(I_tot[i]).item(): .5e}  ')
    if rho.Nv>1 :
        S12, Sx2, Sy2, Sz2 = operators.spin_ddot(rho_0)
        f1.write('      ')
        f1.write(f'{S12: .5e}   {Sx2:.5e}   {Sy2:.5e}   {Sz2:.5e}')
    f1.write(f'    {tdvp.epsilon:3e}  {tdvp.iterator.atol:3e}')
    f1.write(f'    {tdvp.count_sol-1:5d}\n')
    f1.flush()
    f1.close()
    t1 = time.time()

    #tune the epsilon parameter in TDVP
    print(f'timeI: {t1-t0:.3e}')
    tdvp.tune_epsilon()
    tdvp.tune_atol()

    #resample the states
    if sampler.nonmccut < sampler.allcut:
        print(f'tdvp.L2:{tdvp.L2:.4e}')
        if ifstop and tdvp.L2 > 1.:
            np.savetxt('mc_sampling',sampler.samples)
            print(f'something wrong')
            sys.exit()
        for _ in range(5): sampler.flip()
        liouville.set_states_need_mc(rho,sampler.samples)
        tdvp.states_mc = torch.tensor(sampler.samples,device=get_device())

    #save the intermediate RBM
    if count%10==0 and count!=0 :
        torch.save(rho.state_dict(),'para'+str(count))
        print(f'here generate para{count:d}')
```
