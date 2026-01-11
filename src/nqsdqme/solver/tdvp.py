import torch
import numpy as np
import sys
import os
import time
import scipy.optimize as spop
from scipy.special import comb


from ..utils import SubscriptTrans0_torch
from ..core.loggingRK45 import LoggingRK45
from ..global_defs import get_device



# m = self.rho.nstate
# m1 = self.rho.Ns
# n = self.rho.nparameters
# N = states.shape[0]




class TDVP() :
    def __init__(self,basis,rho,liouville,operators,
                 lmbdas = 0.,
                 t_lower=0.,t_upper=30.,
                rtol=1e-13,atol=1e-5,
                k_S=5000,
                ifrk=True,if32=False,
                epsilon=1e-11,min_epsilon=1.e-13,mode='mid'
                ) :
        """
        mode/epsilon selection strategy:
        for steady states, slow, 1e-8/1e-10 is enough? at beginning, then decreasing gradually
        for MCMC, slow or mid; for easy cases, slow or mid, 1e-10 to 1e-14/3/2 
        for difficult cases, quick, 1e-8 to 1e-12
        """
        self.count_sol = 0 #
        self.time_list = [0]
        self.states = torch.tensor(basis.states,device=get_device()) #tensor
        self.rho = rho
        self.liouville = liouville
        self.operator = operators
        self.basis = basis

        self.lmbdas = lmbdas #lambda of deltas^2
        self.t_lower = t_lower
        self.t_upper = t_upper
        # self.t = t_lower
        self.y0 = self.rho.nn_to_vec().to('cpu').numpy()
        self.k_S=k_S
        self.L2 = 0.
        self.deltas2 = 0.
        self.set_epsilon(epsilon=epsilon,min_epsilon=min_epsilon,tuning_mode=mode)
        self.ischolesky = True
        if if32:
            self.rho.grad_dtype = torch.float32
        self.show = True # whether print intermediate message
        if ifrk:
            self.iterator = LoggingRK45(self.solve, t0=self.t_lower, y0=self.y0, t_bound=self.t_upper, rtol=rtol, atol=atol) #1e-7
        

    def set_epsilon(self,epsilon=1e-11,min_epsilon=1.e-13,tuning_mode='mid'):
        self.epsilon = epsilon
        self.tuning_mode = tuning_mode
        if tuning_mode=='slow':
            self.dr = 0.5
            self.dlimit = 1.e-1
        elif tuning_mode=='mid':
            self.dr = 0.2
            self.dlimit = 1.e-1
        elif tuning_mode=='quick':
            self.dr = 0.2
            self.dlimit = 2.e-2
        self.min_epsilon = min_epsilon
        return 0

    def coefficient(self) :
        '''
        calculate S and F in Sx=F, <LdaggerL>, tr(rho0) and trace(rho)
        '''
        rho_0 = torch.zeros(4**self.rho.Ns,dtype=torch.complex128,device=get_device())
        states = self.states
        y = states[:,0:self.rho.Nd]
        gauge_factor = torch.exp(self.lmbdas*torch.sum(y,dim=1)).view(-1,1)
        with torch.no_grad():
            t1 = time.time()
            Llocals = self.liouville.Lforward(self.rho,self.basis.states).view(-1,1)
            t11 = time.time()
            rho_states = self.rho.States(states).view(-1,1)
            t12 = time.time()
        t2 = time.time()
        L2 = torch.sum(torch.real(torch.conj(Llocals)*Llocals)*gauge_factor)
        # print(f'L2:{L2}')
        Z = torch.sum(torch.real(torch.conj(rho_states)*rho_states))

        #calculate S and F in blocks
        t3 = time.time()
        S = torch.zeros((self.rho.nparameters,self.rho.nparameters),dtype=torch.float64,device=get_device())
        F = torch.zeros(self.rho.nparameters,dtype=torch.float64,device=get_device())
        n_states = self.states.shape[0]
        block_size = self.k_S
        n_part = n_states//block_size
        n_remain = n_states%block_size
        left = 0
        for i in range(n_part) :
            right = left + block_size
            gradients = self.rho.compute_batched_grads(states[left:right]).to(torch.complex128)
            F += torch.sum(torch.real(torch.conj(gradients)*Llocals[left:right]*gauge_factor[left:right]),dim=0)
            S += torch.real(torch.matmul(torch.conj(gradients).T,gradients*gauge_factor[left:right]))     
            left = right 
        if n_remain!=0:
            gradients = self.rho.compute_batched_grads(states[left:]).to(torch.complex128)
            F += torch.sum(torch.real(torch.conj(gradients)*Llocals[left:]*gauge_factor[left:]),dim=0)
            S += torch.real(torch.matmul(torch.conj(gradients).T,gradients*gauge_factor[left:]))
        rho_0 = self.rho.rho_0().detach() 
        Z0 = torch.real(torch.trace(rho_0))**2
        t4 = time.time()

        # print(f'time: Ll:{t11-t1:.3e}  states:{t12-t11:.3e}  S+grad:{t4-t3:.3e}',flush=True)
        t_msg = {}
        t_msg['tLlocal'] = t11-t1
        t_msg['tstates'] = t12-t11
        t_msg['tS+grad'] = t4-t3
        self.L2 = L2/Z0
        return S/Z0, F/Z0, L2/Z0, Z, Z0, rho_0, t_msg

    
    # def cholesky(self,S,F,epsilon):
    #     S_e = S + epsilon*torch.eye(S.shape[0],device=get_device())
    #     # Eig = torch.linalg.eigvalsh(S_e)
    #     # print(f'eigan value:{Eig[0:10]}, {Eig[-10:]}')
    #     L = torch.linalg.cholesky(S_e)
    #     y = torch.linalg.solve_triangular(L, F.reshape(-1,1), upper=False)
    #     g = torch.linalg.solve_triangular(L.T, y, upper=True).squeeze() 
    #     if torch.isnan(g).all():
    #         print(
    #             "the matrix S isn't positive definite or is ill-conditioned"+
    #             ", please increase epsilon") 
    #         # self.decreasing = 1.
    #         self.epsilon = self.epsilon*10.
    #     gsg = g.matmul(S.matmul(g)-2*F)
    #     a = S.matmul(g)-F
    #     innerp = a.matmul(g)
    #     delta_equation = a.matmul(a)
    #     b = S_e.matmul(g)-F
    #     delta_e_equation = b.matmul(b)
    #     return g, delta_e_equation, delta_equation, innerp, gsg
    
    def solve_equation(self,S,F,epsilon):
        print(f'sloving Sx=F...',flush=True)
        S_e = S + epsilon*torch.eye(S.shape[0],device=get_device())
        if self.ischolesky:
            L,info = torch.linalg.cholesky_ex(S_e)
            if info!=0:
                print(
                    "the matrix S isn't positive definite or is ill-conditioned"+
                    ", please increase epsilon") 
                # self.decreasing = 1.
                self.epsilon = self.epsilon*10.
                S_e = S + self.epsilon*torch.eye(S.shape[0],device=get_device())
                L,info = torch.linalg.cholesky_ex(S_e)
            y = torch.linalg.solve_triangular(L, F.reshape(-1,1), upper=False)
            g = torch.linalg.solve_triangular(L.T, y, upper=True).squeeze() 
        else:
            #if the S is far from positive definiteness, this step maybe get stuck
            g = torch.linalg.solve(S_e, F) 
        print(f'solved',flush=True)
        gsg = g.matmul(S.matmul(g)-2*F)
        a = S.matmul(g)-F
        innerp = a.matmul(g)
        delta_equation = a.matmul(a)
        b = S_e.matmul(g)-F
        delta_e_equation = b.matmul(b)
        return g, delta_e_equation, delta_equation, innerp, gsg



    def solve(self,t,parameters_numpy) : 
        self.rho.vec_to_nn(torch.from_numpy(parameters_numpy).to(get_device()))
        # self.t = t
        t1 = time.time()
        S,F,L2,Z,Z0,rho_0_torch,t_msg = self.coefficient()
        ldaggerl_unif = L2    
        # t2 = time.time()
        # g = MinresQLP_torch_64(S,F,3000*pow(10,-13),100000)[0]      
        # t3 = time.time()
        t2 = time.time()
        g, delta_e_equation, delta_equation, innerp, gsg = self.solve_equation(S,F,self.epsilon)
        # self.solve_all(S,F,self.epsilon)
        # self.cholesky(S,F,self.epsilon)
        deltas2 = gsg+ldaggerl_unif
        t3 = time.time()

        self.deltas2 = deltas2
        g0 = g.to('cpu').numpy()
        # S0,F0 = (S.to('cpu').numpy()),(F.to('cpu').numpy())
        # t2 = time.time()
        # g0 = MinresQLP(S0,F0,2000*pow(10,-13),100000)[0]
        # t3 = time.time()
        # g = torch.from_numpy(g0).to(get_device())
        n_up, n_down = self.operator.occupation(rho_0_torch)
        trace = torch.real(torch.trace(rho_0_torch))
        self.time_list.append(time.time())

        print(f't: {t:8f}',flush=True)
        print(f'{self.count_sol:d}')
        print(t_msg)
        print(f'epsilon:{self.epsilon}   |A_x-B|^2: {delta_e_equation: .6e}  |Ax-B|^2: {delta_equation: .6e}  deltas: {gsg+ldaggerl_unif: .6e}')
        print(f'x^2:{g.matmul(g): .6e}  <Ax-B,x>:{innerp: .6e}  t: {t3-t2:.4f}  recomendated_minepsilon{delta_e_equation/(2*innerp)}')
        # this recomendation is determined by |A_x-B|^2
        if self.show:
            print(f' trace: {trace:.5e}')
            print(f'deltas: {self.deltas2: .6e}')
            print(f'ldaggl: {ldaggerl_unif: .6e}')
            print(f'  del_: {gsg: .6e}')
            print(f'  norm:  Z: {Z:.6e},  trace**2:{Z0:.6e}')
            print(f'occupy:  up: {np.real(n_up):.5e}  down: {np.real(n_down):.5e}')
            print(f'time:  coefficient: {t2-t1:.4f} , g: {t3-t2:.4f} , all: {self.time_list[-1]-self.time_list[-2]:.3e}')
        print(' ', flush=True)
        f0 = open('t-n_all','a')
        # f0.write(f'{t:.6f}   {n_up:.5e}  {n_down:.5e}    {(gsg+ldaggerl_unif):.5e}  {ldaggerl_unif:.5e}  {self.count_in_fun:5d}\n')
        f0.write(f'{t:.6f}   {n_up:.5e}  {n_down:.5e}   {trace:.3e}   {(gsg+ldaggerl_unif):.5e}  {ldaggerl_unif:.5e}\n')
        f0.flush()
        f0.close()
        
        self.count_sol += 1
        return g0
    
    def tune_epsilon(self):
        print(f'epsilon:{self.epsilon}')
        # self.t = t
        t1 = time.time()
        S,F,L2,Z,Z0,rho_0_torch,t_msg = self.coefficient()
        ldaggerl_unif = L2    
        # t2 = time.time()
        # g = MinresQLP_torch_64(S,F,3000*pow(10,-13),100000)[0]      
        # t3 = time.time()
        t2 = time.time()
        g, delta_e_equation, delta_equation, innerp, gsg = self.solve_equation(S,F,self.epsilon)
        deltas2 = gsg+ldaggerl_unif
        g_1, delta_e_equation_1, delta_equation_1, innerp_1, gsg_1 = self.solve_equation(S,F,self.epsilon*self.dr)
        deltas2_1 = gsg_1+ldaggerl_unif
        t3 = time.time()


        if torch.abs((deltas2_1-deltas2)/deltas2)>self.dlimit and self.epsilon>self.min_epsilon and deltas2>1.e-8:
            # and deltas2>1e-7
            self.epsilon = self.epsilon*self.dr
            g, delta_e_equation, delta_equation, innerp, gsg = \
                g_1, delta_e_equation_1, delta_equation_1, innerp_1, gsg_1
            deltas2 = deltas2_1
        self.deltas2 = deltas2
        print(f'tuned epsilon:{self.epsilon}')
        # print(t_msg)
        print(f'epsilon:{self.epsilon}   |A_x-B|^2: {delta_e_equation: .6e}  |Ax-B|^2: {delta_equation: .6e}  deltas: {gsg+ldaggerl_unif: .6e}')
        print(f'x^2:{g.matmul(g): .6e}  <Ax-B,x>:{innerp: .6e}  t: {t3-t2:.4f}  recomendated_minepsilon{delta_e_equation/(2*innerp)}')
        return self.epsilon

    def tune_atol(self):
        if self.deltas2 < 1e-5 and self.iterator.atol<1e-6:
            self.iterator.atol = 1e-6
        elif self.deltas2 < 1e-7:
            self.iterator.atol = 1e-5
        return self.iterator.atol
    
    def step(self):
        return self.iterator.step()
    
    def LdaggerL(self) :
        '''
        calculate S and F in Sx=F, <LdaggerL>, tr(rho0) and trace(rho)
        '''
        rho_0 = torch.zeros(4**self.rho.Ns,dtype=torch.complex128,device=get_device())
        states = self.states
        y = states[:,0:self.rho.Nd]
        with torch.no_grad():
            t1 = time.time()
            Llocals = self.liouville.Lforward(self.rho,self.basis.states).view(-1,1)
            t11 = time.time()
            rho_states = self.rho.States(states).view(-1,1)
            t12 = time.time()
        t2 = time.time()
        L2 = torch.sum(torch.real(torch.conj(Llocals)*Llocals))
        # print(f'L2:{L2}')
        Z = torch.sum(torch.real(torch.conj(rho_states)*rho_states))
        t_msg = {}
        t_msg['tLlocal'] = t11-t1
        return L2, Z, t_msg

class TDVP_mc(TDVP) :
    def __init__(self, basis, sampler,rho,liouville,operators,
                 lmbdas = 0.,
                  t_lower=0, t_upper=30, 
                  rtol=1e-13, atol=1.e-5, 
                  k_S=5000, ifrk=True, if32=False, 
                  epsilon=1.e-11,min_epsilon=1.e-13,mode='mid',
                 lmbda=-3.,nonmccut=2,allcut=3):
        self.lmbda = lmbda
        self.nonmccut = nonmccut
        self.allcut = allcut
        self.states_mc = torch.tensor(sampler.samples,device=get_device()) #torch.tensor
        self.mc_coefficient = sampler.mc_coefficient
        self.sampler = sampler
        super().__init__(basis, rho,liouville,operators,
                         lmbdas,
                          t_lower, t_upper, 
                          rtol, atol,  k_S, ifrk, if32,epsilon,min_epsilon, mode)

    def coefficient_mc(self) :
        '''
        calculate S and F in Sx=F, <LdaggerL>, tr(rho0) and trace(rho)
        '''
        states = self.states_mc
        N = states.shape[0]
        y = states[:,0:self.rho.Nd]
        weight = torch.exp(self.lmbda*torch.sum(y,dim=1)).view(-1,1)
        gauge_factor = torch.exp(self.lmbdas*torch.sum(y,dim=1)).view(-1,1)
        with torch.no_grad():
            t1 = time.time()
            Llocals = self.liouville.Lforward(self.rho,self.sampler.samples,mode=1).view(-1,1)
            t11 = time.time()
            rho_states = self.rho.States(states).view(-1,1)
            t12 = time.time()
        t2 = time.time()
        L2 = torch.sum(torch.real(torch.conj(Llocals)*Llocals)*gauge_factor/weight)/N
        Z = torch.sum(torch.real(torch.conj(rho_states)*rho_states)/weight)/N

        #calculate S and F in blocks
        t3 = time.time()
        S = torch.zeros((self.rho.nparameters,self.rho.nparameters),dtype=torch.float64,device=get_device())
        F = torch.zeros(self.rho.nparameters,dtype=torch.float64,device=get_device())
        n_states = states.shape[0]
        block_size = self.k_S
        n_part = n_states//block_size
        n_remain = n_states%block_size
        left = 0
        for i in range(n_part) :
            right = left + block_size
            gradients = self.rho.compute_batched_grads(states[left:right]).to(torch.complex128)
            F += torch.sum(torch.real(torch.conj(gradients)*Llocals[left:right])*gauge_factor[left:right]/weight[left:right],dim=0)
            S += torch.real(torch.matmul(torch.conj(gradients).T,gradients*gauge_factor[left:right]/weight[left:right]))     
            left = right 
        if n_remain!=0:
            gradients = self.rho.compute_batched_grads(states[left:]).to(torch.complex128)
            F += torch.sum(torch.real(torch.conj(gradients)*Llocals[left:])*gauge_factor[left:]/weight[left:],dim=0)
            S += torch.real(torch.matmul(torch.conj(gradients).T,gradients*gauge_factor[left:]/weight[left:]))
        S = S/N
        F = F/N
        t4 = time.time()

        # print(f'time: Ll:{t11-t1:.3e}  states:{t12-t11:.3e}  S+grad:{t4-t3:.3e}',flush=True)
        t_msg = {}
        t_msg['tLlocal_mc'] = t11-t1
        t_msg['tstates_mc'] = t12-t11
        t_msg['tS+grad_mc'] = t4-t3
        return S, F, L2, Z, t_msg
        
    def coefficient(self) :
        S_0,F_0,L2_0,Z_0,Z0,rho_0_torch,tmsg = super().coefficient()
        # torch.save([L2_0,Z_0],'LZ_cut'+str(cut))
        # L2_0,Z_0 = torch{{.load('LZ_cut'+str(cut))
        print(f'nonmc,cut={self.nonmccut}:')
        print(f'L2:{L2_0/Z0},  Z:{Z_0}',flush=True)
        print(f'tmsg_nonmc:{tmsg}')
        S_mc,F_mc,L2_mc,Z_mc,tmsg_mc = self.coefficient_mc()

        S = S_0 + S_mc*self.mc_coefficient
        F = F_0 + F_mc*self.mc_coefficient
        L2 = L2_0 + L2_mc*self.mc_coefficient
        Z = Z_0 + Z_mc*self.mc_coefficient
        self.L2 = L2/Z0
        # print(f'self.L2:{self.L2}')
        # print(self.states_mc.shape)
        print(f'mc,cut={self.allcut}:')
        # print(f'mcweight:{self.mc_coefficient}')
        print(f'L2:{L2/Z0},  Z:{Z}',flush=True)
        print(f'tmsg_mc:{tmsg_mc}')
        tmsg.update(tmsg_mc)
        return S/Z0, F/Z0, L2/Z0, Z, Z0, rho_0_torch, tmsg

    def tune_epsilon(self):
        print(f'epsilon:{self.epsilon}')
        # self.t = t
        t1 = time.time()
        S,F,L2,Z,Z0,rho_0_torch,_ = self.coefficient()
        ldaggerl_unif = L2    
        # t2 = time.time()
        # g = MinresQLP_torch_64(S,F,3000*pow(10,-13),100000)[0]      
        # t3 = time.time()
        t2 = time.time()
        g, delta_e_equation, delta_equation, innerp, gsg = self.solve_equation(S,F,self.epsilon)
        deltas2 = gsg+ldaggerl_unif
        g_1, delta_e_equation_1, delta_equation_1, innerp_1, gsg_1 = self.solve_equation(S,F,self.epsilon*self.dr)
        deltas2_1 = gsg_1+ldaggerl_unif
        t3 = time.time()


        if torch.abs((deltas2_1-deltas2)/deltas2)>self.dlimit and self.epsilon>self.min_epsilon and deltas2>1.e-8:
            # and deltas2>1e-7
            self.epsilon = self.epsilon*self.dr
            g, delta_e_equation, delta_equation, innerp, gsg = \
                g_1, delta_e_equation_1, delta_equation_1, innerp_1, gsg_1
            deltas2 = deltas2_1
        self.deltas2 = deltas2
        print(f'tuned epsilon:{self.epsilon}')
        # print(t_msg)
        print(f'epsilon:{self.epsilon}   |A_x-B|^2: {delta_e_equation: .6e}  |Ax-B|^2: {delta_equation: .6e}  deltas: {gsg+ldaggerl_unif: .6e}')
        print(f'x^2:{g.matmul(g): .6e}  <Ax-B,x>:{innerp: .6e}  t: {t3-t2:.4f}  recomendated_minepsilon{delta_e_equation/(2*innerp)}')
        return self.epsilon
    
