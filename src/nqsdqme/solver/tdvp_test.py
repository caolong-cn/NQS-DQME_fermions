import torch
import numpy as np

from ..utils import CUDATimer
from .tdvp import TDVP, TDVP_mc
from ..global_defs import get_device



class TDVP_test(TDVP):
    def __init__(self, basis, rho, liouville, operators,
                 lmbdas = 0.,
                  t_lower=0, t_upper=30,  
                 rtol=1e-13, atol=1e-5,k_S=5000, ifrk=False, if32=False,
                 epsilon=1.e-11, min_epsilon=1.e-13):
        super().__init__(basis, rho,liouville,operators,lmbdas,
                          t_lower, t_upper, rtol, atol,k_S,ifrk,if32,epsilon,min_epsilon,lmbdas)

    def Tblock_size(self,k_S=[100,500,1000,5000,10000]):
        t_S = np.zeros(len(k_S),dtype=np.float64)
        for i in range(len(k_S)):
            self.k_S = k_S[i]
            t_msg = self.coefficient_S_grad()
            print(f'k_S:{k_S[i]}    t:{t_msg}',flush=True)
            t_S[i] = t_msg['tS+grad']
            # t_S[i] = t_msg['tgrad']
        print(f'the best blocksize of S is: {k_S[np.argmin(t_S)]}')
        return t_S
    
    def coefficient_grad(self) :
        '''
        calculate S and F in Sx=F, <LdaggerL>, tr(rho0) and trace(rho)
        '''
        states = self.states
        n_states = self.states.shape[0]

        chunk_size = self.k_S
        with CUDATimer() as timer1:
            for chunk_start in range(0, n_states, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_states)
                gradients = self.rho.compute_batched_grads(states[chunk_start:chunk_end]).to(torch.complex128)
        t_msg = {}
        t_msg['tgrad'] = timer1.get_time_s()
        return t_msg

    def coefficient_S_grad(self,rho) :
        '''
        calculate S and F in Sx=F, <LdaggerL>, tr(rho0) and trace(rho)
        '''
        states = self.states
        n_states = self.states.shape[0]

        S = torch.zeros((self.rho.nparameters,self.rho.nparameters),dtype=torch.float64,device=get_device())
        chunk_size = self.k_S
        with CUDATimer() as timer1:
            for chunk_start in range(0, n_states, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_states)
                gradients = self.rho.compute_batched_grads(states[chunk_start:chunk_end]).to(torch.complex128)
                S += torch.real(torch.matmul(torch.conj(gradients).T,gradients))     
        t_msg = {}
        t_msg['tS+grad'] = timer1.get_time_s()
        return t_msg
    
    def msg_RK45(self,rtol,atol,epsilon):
        self.iterator.rtol = rtol
        self.iterator.atol = atol
        self.epsilon = epsilon
        msg = self.iterator.step()
        print(msg)
        return msg
    
    def Tepsilon(self,epsilon=np.power(10.,np.arange(-6,-17,-2)),show=False):
        self.show = show
        self.decreasing = 1.
        deltas2 = np.zeros_like(epsilon,dtype=np.float64)
        for i in range(epsilon.size):
            self.epsilon = epsilon[i]
            self.solve(self.t_lower,self.rho.nn_to_vec().to('cpu').numpy())
            deltas2[i] = self.deltas2
            if i>0 and np.abs((deltas2[i]-deltas2[i-1])/deltas2[i-1])<1e-2:
                print(f"recomended epsilon:{epsilon[i]}")
                #this recomentation is determined by the convergence of deltas2
        return 0


class TDVP_mc_test(TDVP_mc):
    def __init__(self, basis,sampler, rho,liouville,operators,lmbdas = 0.,
                  t_lower=0, t_upper=30, 
                 rtol=1e-13, atol=1e-5,k_S=5000,ifrk=False,if32=False,
                 epsilon=1.e-11, min_epsilon=1.e-13,
                 lmbda=-3.,nonmccut=2,allcut=3):
        super().__init__(basis,sampler,rho,liouville,operators,lmbdas,
                          t_lower, t_upper, 
                          rtol, atol,k_S,ifrk,if32, epsilon, min_epsilon, 
                         lmbda=lmbda,nonmccut=nonmccut,allcut=allcut)
