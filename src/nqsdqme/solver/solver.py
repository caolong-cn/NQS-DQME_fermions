import torch
import numpy as np
import sys
import os
import time
import scipy.optimize as spop
from scipy.special import comb


from ..utils import SubscriptTrans0_torch
# from init import sys,rho,device,input_t,states,operators
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
    
class SS_pre():
    def __init__(self,allcut,rho,states:torch.Tensor,operators):
        self.rho = rho
        self.states = states
        self.operator = operators
        rho_input = torch.from_numpy(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=(0,1)))
        self.target = (rho_input[:,0] + 1.j*rho_input[:,1]).to(get_device())
        self.coef = 3**torch.sum(states[:,:self.rho.Nd],dim=1,dtype=torch.float64)
        self.N_rho = torch.sum(states[:,:self.rho.Nd],axis=1).to(torch.int64)

        self.meth = 'TNC'#'BFGS'
        self.count = 0
        self.step_p = 100
        self.step_s = 500
        self.count_hopping = 0
        self.t3 = time.time()
        self.t4 = time.time()


    @torch.no_grad()
    def loss(self,vec_of_rho):
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        rho_states = self.rho.States(self.states)
        delta = (rho_states-self.target)*self.coef
        L = torch.real(torch.dot(delta,torch.conj(delta)))
        rho_0 = self.rho.rho_0()
        trace = torch.real(torch.trace(rho_0))
        if self.count%self.step_p==0 :
            # or step_count-step_count_back==1
            self.t4 = time.time()
            print(f'count:{self.count:4d}   loss_{self.meth}:{L:.4e}   t:{self.t4-self.t3:.2e}     trace:{trace}',flush=True)
            self.t3 = time.time()
        # if self.count%min(10,self.count_0)==0:
        #     self.result_print(L,ifresult=False)
        if self.count!=0 and self.count%self.step_s==0:
            torch.save(self.rho.state_dict(),f'rho_{self.meth}_{self.count}')
        self.count += 1
        return L.to('cpu').numpy()

    def loss_d(self,vec_of_rho) :
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        rho_states = self.rho.States(self.states)
        delta = (rho_states-self.target)*self.coef
        L = torch.real(torch.dot(delta,torch.conj(delta)))
        L.backward()
        lossd = self.rho.nnd_to_vec()
        self.rho.zero_grad()
        return lossd.to('cpu').numpy()

    class MyTakeStep:
        '''
        
        '''
        def __init__(self,stepsize):
            self.step = stepsize
            self.rng = np.random.default_rng()

        def __call__(self, x):
            # print('average(para) %f' % np.average(x))

            if np.abs(np.average(x)) > 10:
                print('overflow,then restart\n')
                x = self.rng.uniform(-4,0,x.shape)
                # print('average(new para) %f' % np.average(x))
            x_next = self.unirandom(x)
            while(np.average(x_next) > 5):
                print(f'average over 5:{np.average(x_next)}\n')
                x_next = self.unirandom(x)
            x = x_next
            # self.rho.vec_to_nn(torch.from_numpy(x).to(get_device()))
            return x
        
        def unirandom(self,x) :
            s = self.step
            return x + self.rng.uniform(-s,s,x.shape)


    def print_fun(self,x,f,accepted):
        '''
        print the results of each hopping
        '''
        self.count_hopping += 1
        self.count = 0
        print("step:%2d, accepted %d, loss at minimum %.5e" % (self.count_hopping,int(accepted),f))
        self.rho.vec_to_nn(torch.from_numpy(x).to(get_device()))

        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operator.occupation(rho_0)
        print(f'{self.N_rho[328]:1d} {self.rho.States(self.states[328:329]).item(): .6e}  {self.target[328]: .6e}')
        print(f'{self.N_rho[1270]:1d} {self.rho.States(self.states[1270:1271]).item(): .6e}  {self.target[1270]: .6e}')
        print(f'{self.N_rho[666]:1d} {self.rho.States(self.states[666:667]).item(): .6e}  {self.target[666]: .6e}')
        print(f'{self.N_rho[1460]:1d} {self.rho.States(self.states[1460:1461]).item(): .6e}  {self.target[1460]: .6e}')
        print(f'{self.N_rho[1850]:1d} {self.rho.States(self.states[1850:1851]).item(): .6e}  {self.target[1850]: .6e}')
        print(f'{self.N_rho[620]:1d} {self.rho.States(self.states[620:621]).item(): .6e}  {self.target[620]: .6e}')
        print(f'up:{n_up:6e} ,down:{n_down:6e} ,trace: {torch.real(torch.trace(rho_0)):.5e}')

        print('loss = %.5e\n' % (self.loss(x)))
        torch.save(self.rho.state_dict(),f'rho_simulate_{self.meth}_{self.count_hopping}')
        print('\n')
        return 0
    
    def hopping(self,niter=5,T=0.001,step_h=0.65,tol=1e-10,maxfun=10000,step_p=1000,step_s=5000):
        #tnc求解全局最小值
        #主要由'maxfun'：Max. number of function evaluations来调控每一次优化，一般设置为10000-15000即可
        #可优化至约1e-2量级
        #tnc优化必须是float64
        mytakestep = self.MyTakeStep(step_h)
        self.meth = 'TNC'
        self.step_p = step_p
        self.step_s = step_s
        result = spop.basinhopping(self.loss,self.rho.nn_to_vec(),niter=niter,T=T,
            take_step=mytakestep,callback=self.print_fun,
            minimizer_kwargs={'method':self.meth,'jac':self.loss_d,'tol':tol,'options':{'maxfun':maxfun}})
        self.result_print(self.loss(result.x),result)
        return result

    def optimization(self,meth='BFGS',tol=1e-8,step_p=50,step_s=200):
        self.meth = meth
        print(f'tol:{tol}')
        print(f'method:{meth}')
        maxiter = 80*len(self.rho.nn_to_vec())
        self.count = 0
        self.step_p = step_p
        self.step_s = step_s
        if meth=='L-BFGS-B':
            ftol = 1e-15
            result = spop.minimize(self.loss,self.rho.nn_to_vec(),method=meth,jac=self.loss_d,tol=tol,options={'maxiter':maxiter,'ftol':ftol})
        else:
            result = spop.minimize(self.loss,self.rho.nn_to_vec(),method=meth,jac=self.loss_d,tol=tol,options={'maxiter':maxiter})
        self.result_print(self.loss(result.x),result)
        torch.save(self.rho.state_dict(),'rho_BFGS_'+str(self.rho.Nh))

        rho_0 = self.rho.rho_0().detach()
        print(f'origin_trace: {torch.real(torch.trace(rho_0)): .3e}\n',flush=True)
        print(f'rho_0: {rho_0}')
        return 0 
      
    def result_print(self,loss,result):
        '''
        print the results of the total basinhopping
        result is the returned value of spop.basinhopping
        '''
        print("n_it: %d" % result.nit)
        print("total evaluations: %d" % result.nfev,flush=True)
        print(f'terminate because:{result.message}')
        self.rho.vec_to_nn(torch.from_numpy(result.x).to(get_device()))

        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operator.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))
        print('rho_0:')
        print(rho_0)
        print(f'up:{n_up:6e}  ;down:{n_down:6e}')
        print(f'trace: {trace:.3e}')
        print('loss_at_end')
        print(f'loss:{loss:.5e}')
        return 0

class SS_pre_troch(SS_pre):
    def __init__(self, allcut,rho,states,operators,lr,scheduler):
        super().__init__(allcut,rho,states,operators)
        self.lr = lr
        self.scheduler = scheduler

    def loss(self):
        rho_states = self.rho.States(self.states)
        delta = (rho_states-self.target)*self.coef
        L = torch.real(torch.dot(delta,torch.conj(delta)))
        return L

    def loss_d(self) :
        pass

    def optimization(self,meth='Adam',lr=3e-5):
        def closure():
            optimizer.zero_grad()
            L = self.loss()
            L.backward()
            return L
        # t1 = time.time()
        # optimizer.step(closure)
        # t2 = time.time()
        # print(t2-t1)

        optimizer = torch.optim.Adam(self.rho.parameters(),lr=lr)
        sche = torch.optim.lr_scheduler.StepLR(optimizer, 
                step_size=3000, gamma=0.8, last_epoch=- 1, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer, 
        #             mode='min', 
        #             factor=0.8, 
        #             patience = 2, 
        #             verbose = True, 
        #             threshold = 1e-5, 
        #             threshold_mode='rel', 
        #             cooldown=0, 
        #             min_lr=1e-08, 
        #             eps=1e-06
        #         )
        # t1 = time.time()
        # optimizer.step(closure)
        # t2 = time.time()
        # print(t2-t1)
        # sys_self.liouville.exit()
        i = 0
        L = self.loss()
        rho_0 = self.rho.rho_0().detach()
        trace = torch.real(torch.trace(rho_0))
        print(f"count:{i:4d}   ll:{L:.4e}       trace:{trace}       lr:{optimizer.state_dict()['param_groups'][0]['lr']}",flush=True)
        t1 = time.time()
        while L>1e-5:
            i += 1
            optimizer.zero_grad()
            L = self.loss()
            L.backward()
            optimizer.step()
            if i%50==0:
                t2 = time.time()
                rho_0 = self.rho.rho_0().detach()
                trace = torch.real(torch.trace(rho_0))
                print(f"count:{i:4d}   ll:{L:.4e}   t:{t2-t1:.2e}    trace:{trace}      lr:{optimizer.state_dict()['param_groups'][0]['lr']}",flush=True)
                t1 = t2
            if i%5000==0:
                torch.save(self.rho.state_dict(),f'rho_ll_{i}')
        return 0  


    def print_physics(self,operators):
        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operator.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))
        I = self.operator.current_general(self.rho,rho_0)
        Ib = torch.sum(I,dim=0)
        I_tot = torch.sum(Ib).reshape(1)
        f1 = open('result_ll','a')
        f1.write(f'{self.count:4d}   {n_up:.5e}  {n_down:.5e}   {trace:.3e}      ')
        if self.rho.Nb == 1 : #self.rho.Nb=1: tt, j_left_u, j_left_d, j_left_u+j_left_d
            f1.write(f'{torch.real(I[0][0]).item(): .5e}  {torch.real(I[1][0]).item(): .5e}  {torch.real(I_tot[0]).item(): .5e}')
        else : #self.rho.Nb>1: tt, j_left, j_left, j_left+j_left
            I_tot = torch.cat((Ib,I_tot))
            for i in range(self.rho.Nb+1) : 
                f1.write(f'{torch.real(I_tot[i]).item(): .5e}  ')
        if self.rho.Nv>1 :
            S12, Sx2, Sy2, Sz2 = self.operator.spin_ddot(rho_0)
            f1.write('      ')
            f1.write(f'{S12: .5e}   {Sx2:.5e}  {Sy2:.5e}  {Sz2:.5e}')
        f1.write('\n')
        f1.flush()
        f1.close()
        return 0




class SS_ll(SS_pre):
    def __init__(self, allcut,rho,liouville,states,operators,w_l=1.,w_tr=2.):
        super().__init__(allcut,rho,states,operators)
        self.liouville  = liouville
        self.w_l = w_l
        self.w_tr = w_tr
    
    @torch.no_grad()
    def loss(self,vec_of_rho):
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        Lforward = self.liouville.Lforward(self.rho,self.states.to('cpu').numpy())
        L2 = torch.sum(torch.real(torch.multiply(Lforward,Lforward.conj())))
        rho_0 = self.rho.rho_0()
        trace = torch.real(torch.trace(rho_0))
        L = self.w_l*L2/trace**2 + self.w_tr*(trace-1)**2
        if self.count%self.step_p==0 :
            # or step_count-step_count_back==1
            self.t4 = time.time()
            print(f'count:{self.count:4d}   ll_{self.meth}:{L:.4e}   t:{self.t4-self.t3:.2e}     trace:{trace}',flush=True)
            self.t3 = time.time()
        # if self.count%min(10,self.count_0)==0:
        #     self.result_print(L,ifresult=False)
        if self.count!=0 and self.count%self.step_s==0:
            torch.save(self.rho.state_dict(),f'rho_{self.meth}_{self.count}')
        self.count += 1
        return L.to('cpu').numpy()
    
    def loss_d(self,vec_of_rho) :
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        Lforward = self.liouville.Lforward(self.rho,self.states.to('cpu').numpy())
        L2 = torch.sum(torch.real(torch.multiply(Lforward,Lforward.conj())))
        rho_0 = self.rho.rho_0()
        trace = torch.real(torch.trace(rho_0))
        L = self.w_l*L2/trace**2 + self.w_tr*(trace-1)**2
        L.backward()
        lossd = self.rho.nnd_to_vec()
        self.rho.zero_grad()
        return lossd.to('cpu').numpy()
    
    def optimization(self,meth='BFGS',tol=1e-8,step_p=20,step_s=00):
        self.meth = meth
        print(f'tol:{tol}')
        print(f'method:{meth}')
        maxiter = 80*len(self.rho.nn_to_vec())
        self.count = 0
        self.step_p = step_p
        self.step_s = step_s
        if meth=='L-BFGS-B':
            ftol = 1e-15
            result = spop.minimize(self.loss,self.rho.nn_to_vec(),method=meth,jac=self.loss_d,tol=tol,options={'maxiter':maxiter,'ftol':ftol})
        else:
            result = spop.minimize(self.loss,self.rho.nn_to_vec(),method=meth,jac=self.loss_d,tol=tol,options={'maxiter':maxiter})
        self.result_print(self.loss(result.x),result)
        torch.save(self.rho.state_dict(),'rho_BFGS_'+str(self.rho.nhidden))

        rho_0 = self.rho.rho_0().detach()
        print(f'origin_trace: {torch.real(torch.trace(rho_0)): .3e}\n',flush=True)
        print(f'rho_0: {rho_0}')
        return 0