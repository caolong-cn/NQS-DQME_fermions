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
# from init_np import sys,rho,device,states,allcut


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
input_t = np.loadtxt("input",skiprows=4,dtype=np.float64)
nhidden = input_para[0,0]
# nauxillary = input_para[0,1]
nonmccut = input_para[1,0]
allcut = input_para[1,1]


method = sys.argv[1]
if method=='bfgs':
    liouville = LiouvilletSaveL(ReadHamilton(),get_device(),nonmccut,allcut)
elif method=='adam':
    liouville = Liouvillet(ReadHamilton(),get_device(),nonmccut,allcut)

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
rho = MLPt(liouville.nvar,liouville.nspin,liouville.ncor,liouville.nalf
           ,liouville.nsgn,get_device(),
           table0,N_t=1,gamma=-0.,nhidden=nhidden).to(get_device())

        # N_t=3,
        # f_t=[f_1,f_2,f_3],
        # gamma=-2.,
        # nhidden=30
#initialize the basis and sampler
basis = Basis(liouville,nonmccut,allcut)

sampler = Sampler(basis,0,
                 nonmccut,allcut,
                 lmbda=-3.)


print('states_exact',':',basis.states.shape)
print(f'nhidden:{rho.nhidden:d}')
print(f'number of parameters:{rho.nparameters:d}',flush=True)
print(f'gamma:{rho.gamma}')
t0 = time.time()
# liouville.set_states_0(basis.states)
t1 = time.time()
if method=='bfgs':
    liouville.set_states_need(rho,basis.states)

states_torch = torch.tensor(basis.states,device=get_device())
operators = Operators(liouville,states_torch)
coef = 3**torch.sum(states_torch[:,:rho.Nd],dim=1,dtype=torch.float64)


rho_name = 'rho_BFGS_ll_t_900'
rho_name = 'rho_BFGS_ll_tr_4.5_0.1'
print(rho_name)
# rho_intial = MLPt(1,2,4,2,2,70,device,rho.table0).to(device)
# rho_intial.load_state_dict(torch.load('rho_initial',map_location=device))
rho.load_state_dict(torch.load(rho_name,map_location=get_device())) 
# rho.t_to_0()
# a = rho.nn_to_vec()
# rho.vec_to_nn(a*(1+np.random.uniform(-0.01,0.01,a.shape)))
w_b = 0.8
w_e = 1.-w_b
w_r = 20.
delta_t = 1e-8

# print(rho.nn[0].weight)
# print(rho.nn[0].weight[:,-1])
# rho.nn[0].weight[:,-1] = 0. + 0.j
# rho.t_to_0_random()

t_0 = 4.5
t_end = 6.
t_step = 0.2
residual_ts = torch.arange(t_0,t_end,t_step,dtype=torch.float64,device=get_device())
print(f'ts:{residual_ts}')
#loading rho and target
print(f't_0:{t_0}')
if t_0<10.:
    rho_input = torch.from_numpy(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=(0,1)))
    target = (rho_input[:,0] + 1.j*rho_input[:,1]).to(get_device())
else:
    rho1 = copy.deepcopy(rho)
    rho1.load_state_dict(torch.load(rho_name,map_location=device))
    # 'rho_t_'+str(rho.nhidden)+'_4._0.5'
    target = rho1.States(states_torch,residual_ts[0]).detach()
# t_shape = torch.ones_like(sys.states_need['states_need'][:,-1],dtype=torch.float64,device=device)


rho_0_target = torch.zeros(4**rho.Ns,dtype=torch.complex128,device=get_device())
for i in range(2*2**rho.Ns) :
    if torch.sum(states_torch[i,0:rho.Nd])==0 and (states_torch[i,rho.Nd:rho.Nd+rho.Ns]==states_torch[i,rho.Nd+rho.Ns:]).all() :
        position = SubscriptTrans0_torch(states_torch[i])
        rho_0_target[int(position)] = target[i]
rho_0_target = rho_0_target.detach().reshape(2**rho.Ns,2**rho.Ns)
n_up_target, n_down_target = operators.occupation(rho_0_target)

print(f'up_target:{n_up_target:6e}  ;down_target:{n_down_target:6e}')


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
    solver = PINNSolverAdam(liouville,
                    operators,
                    basis,residual_ts,rho,
                    coef,target,
                    w_b ,w_e,w_r,delta_t
                    )
    print(f'loss:{solver.loss(solver.states,solver.target_initial,solver.coef_initial)}')
    print(f'loss:{solver.loss(solver.states,solver.target_initial,solver.coef_initial)}')

    solver.solve(1000)

# sys.exit()



#nohup python3 ./main_t_bfgs.py bfgs &
#nohup python3 ./main_t_bfgs.py adam &