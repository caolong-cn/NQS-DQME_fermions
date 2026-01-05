import torch
import numpy as np
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from nqsdqme.state.NADO import RBM
from nqsdqme.core.liouville import LiouvilleSaveL
from nqsdqme.core.read import ReadHamilton
from nqsdqme.core.operators import Operators

from nqsdqme.core.basis import Basis
from nqsdqme.core.sampler import Sampler

from nqsdqme.solver import SS_pre, SS_ll
from nqsdqme.utils import SubscriptTrans0_torch
from nqsdqme.global_defs import update_device,get_device



#input文件
#cuda/cpu
#hidden auxillary 0
#nonmccut allcut 0(>nonmccut <=allcut的做mc)  (nonmccut与allcut按照Nrho算)
#chains Nstart Nend
#t0 count

#initialize hamilton and device
input_device = np.loadtxt("input",max_rows=1,dtype=np.str_)
device = torch.device(str(input_device))
update_device(device)
print(f'device:{device}')


#initialize hyperparameters and define the problem to be solved  
input_para = np.loadtxt("input",skiprows=1,max_rows=3,dtype=np.int32)
input_t = np.loadtxt("input",skiprows=4,max_rows=1,dtype=np.float64)
nhidden = input_para[0,0]
nauxillary = input_para[0,1]
epsilonmode = input_para[0,2]
nonmccut = input_para[1,0]
allcut = input_para[1,1]
notsaving = input_para[1,2] # = 0(1): (not) saving L
# chains_number = input_para[2,0]
# N_start = input_para[2,1]
# N_end = input_para[2,2]
mc_size = input_para[2,0]

liouville = LiouvilleSaveL(ReadHamilton(),get_device(),nonmccut,allcut,notsaving,mc_size=mc_size)
if nonmccut<allcut:
    liouville.lmbda = -3.
nstate = 2*liouville.nvar*liouville.nspin + liouville.nsgn*liouville.nvar*liouville.nspin*liouville.ncor*liouville.nalf
Ns = liouville.nvar * liouville.nspin
Nd = liouville.nsgn * Ns * liouville.nalf * liouville.ncor
statescols = [i for i in range(3,3+nstate)]
filename="table_cut"+str(nonmccut+1)+".data"
table = np.loadtxt("table_cut3.data",usecols=(statescols),dtype=np.int8)
#table0 consists of the non-zero states of rdo
condition = np.sum(table[:,:Nd],axis=1)==0
table0 = table[condition][:,Nd:]

#initialize the nn
rho = RBM(liouville.nvar,liouville.nspin,liouville.ncor,liouville.nalf,liouville.nsgn,nhidden,nauxillary,get_device(),table0).to(get_device())
# #load the nn
# count = int(input_t[2])
# rho.load_state_dict(torch.load('para'+str(count),map_location=get_device())) 

#initialize the basis and sampler
basis = Basis(liouville,nonmccut,allcut)

sampler = Sampler(basis,mc_size,
                 nonmccut,allcut,
                 lmbda=-3.)


print('states_exact',':',basis.states.shape)
print(f'nhidden:{rho.Nh:d}, nauxillary:{rho.Na:d}')
print(f'number of parameters:{rho.nparameters:d}',flush=True)
t0 = time.time()
# liouville.set_states_0(basis.states)
t1 = time.time()
if notsaving==0 :
    liouville.set_states_need(rho,basis.states)
    if nonmccut<allcut:
        liouville.set_states_need_mc(rho,sampler.samples)

        print(f'time of mc intialization:{t1-t0}')
        print(f'sampling number:{sampler.N_samples}')

states_torch = torch.tensor(basis.states,device=get_device())
operators = Operators(liouville,states_torch)

# N = states.shape[0]

#initialize the TDVP solver
input_epsilon = np.loadtxt("input",skiprows=5,dtype=np.float64)

t_lower = input_t[0]
t_upper = input_t[1] #200

# solver = SS_pre(allcut,rho,states_torch,operators)
solver = SS_ll(allcut,rho,liouville,states_torch,operators)

rho_0_target = torch.zeros(4**rho.Ns,dtype=torch.complex128,device=device)
for i in range(2*2**rho.Ns) :
    if torch.sum(states_torch[i,0:rho.Nd])==0 and (states_torch[i,rho.Nd:rho.Nd+rho.Ns]==states_torch[i,rho.Nd+rho.Ns:]).all() :
        position = SubscriptTrans0_torch(states_torch[i]).to(torch.int8)
        rho_0_target[position] = solver.target[i]
rho_0_target = rho_0_target.detach().reshape(2**rho.Ns,2**rho.Ns)
n_up_target, n_down_target = operators.occupation(rho_0_target)
print(f'up_target:{n_up_target:6e}  ;down_target:{n_down_target:6e}')

print(rho.bd.device)
solver.hopping(niter=3,T=0.001,step_h=0.65,
               tol=1e-10,maxfun=10000,step_p=1000,step_s=5000)
solver.optimization(meth='BFGS',tol=1e-8,step_p=100,step_s=500)


