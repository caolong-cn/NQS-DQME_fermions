import torch
import numpy as np
import sys
import time
import os

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

# if nonmccut < allcut:
#     if not os.path.exists('state0_mc') :
#         statescols = [i for i in range(3,3+nstate)]
#         table = torch.from_numpy(np.loadtxt("table_cut"+str(nonmccut+1)+".data",usecols=statescols)).to(torch.int8)
#         i = np.random.randint(4000,10000)
#         state0 = table[i]
#         torch.save(state0,'state0_mc')
#     lamda = -3.
#     chainsampler = Chainsampler(rule='mhmc',N_start=N_start,N_end=N_end,cut1=nonmccut,cut2=allcut,
#         lmbda=lamda,choices=chains_number,nstate=nstate)
#     mcfile = 'sampling_results_'+str(N_end-N_start)+'_mccut'+str(nonmccut)
#     if os.path.exists(mcfile) :
#         states_mc = torch.load(mcfile)
#     else:
#         states_mc = chainsampler.sampling(rho)
#         torch.save(states_mc,mcfile)
#     print('states_mc',':',states_mc.shape)

states_torch = torch.tensor(basis.states,device=get_device())
operators = Operators(liouville,states_torch)







count = int(input_t[2])
rho.load_state_dict(torch.load('para'+str(count),map_location=get_device())) 


m = rho.nstate
m1 = rho.Ns
n = rho.nparameters
# N = states.shape[0]

input_epsilon = np.loadtxt("input",skiprows=5,dtype=np.float64)


t_lower = input_t[0]
t_upper = input_t[1] #200
y0 = (rho.nn_to_vec()).to('cpu').numpy()


if len(sys.argv)>1:
    tdvp = TDVP_test(basis,rho,liouville,operators,t_lower=t_lower,t_upper=t_upper,atol=1e-7,epsilon=input_epsilon[0],min_epsilon=input_epsilon[1])
    if sys.argv[1]=='Te':
        tdvp.Tepsilon(epsilon=np.power(10.,np.arange(-6,-17,-2)))
    elif sys.argv[1]=='Tb':
        tdvp.Tblock_size(rho,k_S=[16000,17000,20000,30466,31000])
    elif sys.argv[1]=='pL':
        #print L2
        L2,Z,t_msg = tdvp.LdaggerL()
        print(f'L2:{L2:.8e}')
    elif sys.argv[1]=='pLnew':
        #print L2
        L2,Z,t_msg = tdvp.LdaggerL()
        print(f'L2:{L2:.8e}')
    else:
            #print L2_mc
            t0 = time.time()
            for i in range(100):
                sampler.flip()
            t1 = time.time()
            print(f'time of intial flip:{t1-t0}')
            print(f'sampling number :{sampler.N_samples}')
            if sys.argv[1]=='testmc':
                sampler.samples = np.loadtxt('mc_sampling')
                # condition = np.sum(liouville.states_mc[0:rho.Nd],axis=1)>3
                # print(np.sum(liouville.states_mc[0:rho.Nd],axis=1))
            liouville.set_states_need_mc(rho,sampler.samples)
            states_mc = torch.tensor(sampler.samples,device=get_device())
            tdvp = TDVP_mc_test(basis,sampler,rho,liouville,operators,
                                t_lower=t_lower,t_upper=t_upper,
                    atol=1e-5,k_S=10000,
                    epsilon=input_epsilon[0],min_epsilon=input_epsilon[1],
                    lmbda=liouville.lmbda,nonmccut=liouville.nonmccut,allcut=liouville.allcut)
            S,F,L2,Z,Z0,rho_0_torch,tmsg = tdvp.coefficient()
            print(f'L2:{L2:.8e}')
    sys.exit()
else:
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
    # tdvp = TDVP(states,rho,t_lower=t_lower,t_upper=t_upper,epsilon=1.e-12,atol=1.e-5,k_g=200,k_S=5000)

# tdvp = TDVP_test(states,rho,t_lower=t_lower,t_upper=t_upper,epsilon=1e-14,atol=1e-7)
# tdvp.Tblock_size(rho,rho32,liouville)
# tdvp.Tblock_size(rho,rho32,liouville,k_g=[400,500,600],k_S=[10000])
# tdvp.Tepsilon(epsilon=np.power(10.,np.arange(-6,-10,-2)))
# sys.exit()
# k_S=[100,],k_g=[]
# S,F,L2,Z,Z0,rho_0_torch,t_msg = tdvp.coefficient(rho,rho32,sys)
# print(f'L2:{L2},  Z:{Z}')
# sys.exit()
# fun.fun(t_lower,y0)
tdvp.ischolesky = True
# False
ifstop = False #determine whether still evolution when L2>1
while tdvp.iterator.t < t_upper :
    msg = tdvp.iterator.step() #important
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
    print(f'timeI: {t1-t0:.3e}')
    tdvp.tune_epsilon()
    tdvp.tune_atol()
    if sampler.nonmccut < sampler.allcut:
        print(f'tdvp.L2:{tdvp.L2:.4e}')
        if ifstop and tdvp.L2 > 1.:
            np.savetxt('mc_sampling',sampler.samples)
            print(f'something wrong')
            sys.exit()
        for _ in range(5): sampler.flip()
        liouville.set_states_need_mc(rho,sampler.samples)
        tdvp.states_mc = torch.tensor(sampler.samples,device=get_device())


    if count%10==0 and count!=0 :
        torch.save(rho.state_dict(),'para'+str(count))
        print(f'here generate para{count:d}')
