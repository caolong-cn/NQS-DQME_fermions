import torch
import numpy as np
import sys
import time

import os

import scipy.optimize as spop

from ..state import NADOt
from ..core.liouville import Liouville
from nqsdqme.core.basis import Basis

from ..global_defs import get_device

class PINNSolver():
    def __init__(self,liouville:Liouville,
                 operators,
                 basis:Basis,residual_ts:np.ndarray,rho0:NADOt,
                 coef_initial,target_initial,
                 w_b = 0.8,w_e = 0.2,w_r = 20.,delta_t = 1e-8,
                 ):
        self.basis = basis
        self.states = torch.tensor(basis.states,device=get_device())
        # self.states.shape[0]

        self.liouville = liouville
        self.operators = operators
        self.rho = rho0
        self.residual_ts = residual_ts
        self.N_r = residual_ts.numel()
        self.delta_t = delta_t
        self.vec_of_rho = self.rho.nn_to_vec()

        self.coef_initial = coef_initial
        self.target_initial = target_initial

        self.w_b = w_b
        self.w_e = w_e
        self.w_r = w_r

        self.L2 = 0.



class PINNSolverSPOP(PINNSolver):
    def __init__(self,liouville:Liouville,
                 operators,
                 basis:Basis,residual_ts:np.ndarray,rho0:NADOt,
                 coef_initial,target_initial,
                 w_b = 0.8,w_e = 0.2,w_r = 20.,delta_t = 1e-8,
                 method='BFGS'
                 ):
        super().__init__(liouville,
                 operators,
                 basis,residual_ts,rho0,
                 coef_initial,target_initial,
                 w_b ,w_e,w_r,delta_t)

        self.method = method
        self.count_loss = 0 #count the call times of self.loss
        self.stepsize_print = 10 #the stepsize of print loss
        self.stepsize_saverho = 100
        self.t1 = time.time()
        self.t2 = time.time()
        # self.t_input = torch.zeros((residual_ts.shape[0],self.rho.N_t),dtype=torch.float64,device=get_device())
        # self.t_input_dt = torch.zeros((residual_ts.shape[0],self.rho.N_t),dtype=torch.float64,device=get_device())
        # for i in range(self.rho.N_t):
        #     self.t_input[:,i] = self.rho.f_t[i](residual_ts)
        #     self.t_input_dt[:,i] = self.rho.f_t[i](residual_ts+self.delta_t)

    @torch.no_grad()
    def loss(self,vec_of_rho) : 
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        #contribution of initial conditions/ boundary conditions
        t_0 = self.residual_ts[0]
        rho_states = self.rho.States(self.states,t_0)
        delta = (rho_states-self.target_initial)*self.coef_initial
        L_b = torch.real(torch.dot(delta,torch.conj(delta))) 

        #contribution of equation
        L_e = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        L_tr = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        for i in range(0,self.N_r):
            self.residual_ts[i]
            a = self.rho.States(self.states,self.residual_ts[i])
            b = self.rho.States(self.states,self.residual_ts[i]+self.delta_t)
            partial_t_of_rho = (b - a)/self.delta_t
            residual = partial_t_of_rho - self.liouville.Lforward(self.rho,self.residual_ts[i])
            # Lforward_new_nonmc_2(rho)
            # print(f'partial_t_of_rho:{torch.sum(torch.real(torch.multiply(partial_t_of_rho,partial_t_of_rho.conj())))}')
            L2 = torch.sum(torch.real(torch.multiply(residual,residual.conj())))
            trace    = torch.real(torch.sum(a[0:2**self.rho.Ns]))
            L_e[i] =  L2/(trace**2)
            L_tr[i] = (1-trace)**2
        loss = self.w_b*L_b + self.w_e*torch.sum(L_e) +self.w_r*torch.sum(L_tr)
        if self.count_loss%self.stepsize_print==0 :
            # if self.count==0:
            #     t3 = time.time()
            self.t2 = time.time()
            print(f'count:{self.count_loss:4d}   \
                  loss_{self.method}:{loss:.4e}   t:{self.t2-self.t1:.2e} \
                    trace_at_last:{trace}',flush=True)
            self.save_physics(loss)
            self.t1 = time.time()
        # if self.count_loss%self.stepsize_print==0:
            # result_print(loss,ifresult=False)
        if (self.count_loss!=0 and 
            self.count_loss%self.stepsize_saverho==0):
            # and meth=='BFGS':
            torch.save(self.rho.state_dict(),f'rho_'+self.method+f'_ll_t_{self.count_loss}')
        self.count_loss += 1
        self.L2 = loss.to('cpu').numpy()
        return self.L2

    def loss_d(self,vec_of_rho) :
        self.rho.vec_to_nn(torch.from_numpy(vec_of_rho).to(get_device()))
        #contribution of initial conditions/ boundary conditions
        t_0 = self.residual_ts[0]
        rho_states = self.rho.States(self.states,t_0)
        delta = (rho_states-self.target_initial)*self.coef_initial
        L_b = torch.real(torch.dot(delta,torch.conj(delta))) 

        #contribution of equation
        L_e = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        L_tr = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        for i in range(0,self.N_r):
            self.residual_ts[i]
            a = self.rho.States(self.states,self.residual_ts[i])
            b = self.rho.States(self.states,self.residual_ts[i]+self.delta_t)
            partial_t_of_rho = (b - a)/self.delta_t
            residual = partial_t_of_rho - self.liouville.Lforward(self.rho,self.residual_ts[i])
            L2 = torch.sum(torch.real(torch.multiply(residual,residual.conj())))
            trace    = torch.real(torch.sum(a[0:2**self.rho.Ns]))
            L_e[i] =  L2/(trace**2)
            L_tr[i] = (1-trace)**2
        loss = self.w_b*L_b + self.w_e*torch.sum(L_e) +self.w_r*torch.sum(L_tr)
        loss.backward()
        lossd = self.rho.nnd_to_vec()
        self.rho.zero_grad()
        return lossd.to('cpu').numpy()
    
    def solve(self,tol=1e-8):
        if self.method=='BFGS':
            maxiter = 80*len(self.rho.nn_to_vec())
            result = spop.minimize(
                self.loss,self.vec_of_rho,
                method=self.method,jac=self.loss_d,
                tol=tol,options={'maxiter':maxiter})
        return result
    
    def save_physics(self,loss):
        #ifresult=True: prints the info of optimization after the whole BFGS optimization
        #ifresult=Fale: prints the info 

        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operators.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))

        I = self.operators.current_general(self.rho,rho_0)
        Ib = torch.sum(I,dim=0)
        I_tot = torch.sum(Ib).reshape(1)
        f1 = open('result_ll','a')
        f1.write(f'{self.count_loss:4d}   {n_up:.5e}  {n_down:.5e}   {trace:.3e}   {loss:.5e}      ')
        if self.liouville.nalf == 1 : #rho.Nb=1: tt, j_left_u, j_left_d, j_left_u+j_left_d
            f1.write(f'{torch.real(I[0][0]).item(): .5e}  {torch.real(I[1][0]).item(): .5e}  {torch.real(I_tot[0]).item(): .5e}')
        else : #rho.Nb>1: tt, j_left, j_left, j_left+j_left
            I_tot = torch.cat((Ib,I_tot))
            for i in range(self.liouville.nalf+1) : 
                f1.write(f'{torch.real(I_tot[i]).item(): .5e}  ')
        if self.rho.Nv>1 :
            S12, Sx2, Sy2, Sz2 = self.operators.spin_ddot(rho_0)
            f1.write('      ')
            f1.write(f'{S12: .5e}   {Sx2:.5e}  {Sy2:.5e}  {Sz2:.5e}')
        f1.write('\n')
        f1.flush()
        f1.close()

    def print_result(self,result):
        #prints the info of the whole BFGS optimization
        print("n_it: %d" % result.nit)
        print("total evaluations: %d" % result.nfev,flush=True)
        print(f'terminate because:{result.message}')
        self.rho.vec_to_nn(torch.from_numpy(result.x))

        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operators.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))
        print('rho_0:')
        print(f'up:{n_up:6e}  ;down:{n_down:6e}')
        print(f'trace: {trace:.3e}')
        print('loss_at_end')
        print(f'loss:{self.L2:.5e}')



class PINNSolverAdam(PINNSolver):
    def __init__(self,liouville:Liouville,
                 operators,
                 basis:Basis,residual_ts:np.ndarray,rho0:NADOt,
                 coef_initial,target_initial,
                 w_b = 0.8,w_e = 0.2,w_r = 20.,delta_t = 1e-8,
                 method='Adam',batch_size=256
                 ):
        super().__init__(liouville,
                 operators,
                 basis,residual_ts,rho0,
                 coef_initial,target_initial,
                 w_b ,w_e,w_r,delta_t)
        
        self.method = 'Adam'
        self.count_epoch = 0 #count the call times of self.loss
        self.stepsize_print = 10 #the stepsize of print loss
        self.stepsize_saverho = 100
        self.t1 = time.time()
        self.t2 = time.time()

        self.batch_size = batch_size

        # self.stepsize_print = 200
        # self.stepsize_saverho = 5000

    def loss(self,states:torch.Tensor,targets_initial:torch.Tensor,coef_initial:torch.Tensor) : 
        #contribution of initial conditions/ boundary conditions
        t_0 = self.residual_ts[0]
        rho_states = self.rho.States(states,t_0)
        delta = (rho_states-targets_initial)*coef_initial
        L_b = torch.real(torch.dot(delta,torch.conj(delta))) 

        #contribution of equation
        L_e = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        L_tr = torch.zeros_like(self.residual_ts,dtype=torch.float64,device=get_device())
        for i in range(0,self.N_r):
            a = self.rho.States(states,self.residual_ts[i])
            b = self.rho.States(states,self.residual_ts[i]+self.delta_t)
            partial_t_of_rho = (b - a)/self.delta_t
            residual =  partial_t_of_rho - self.liouville.Lforward(self.rho,states,t=self.residual_ts[i])
            # Lforward_new_nonmc_2(rho)
            # print(f'partial_t_of_rho:{torch.sum(torch.real(torch.multiply(partial_t_of_rho,partial_t_of_rho.conj())))}')
            L2 = torch.sum(torch.real(torch.multiply(residual,residual.conj())))
            rho0 = self.rho.States(self.states[0:2**self.rho.Ns],self.residual_ts[i])
            trace  = torch.real(torch.sum(rho0))
            L_e[i] =  L2/(trace**2)
            L_tr[i] = ((1-trace)**2)*states.shape[0]/self.states.shape[0]
        # print(f'L_e:{L_e}')
        # print(f'L_b:{L_b}')
        # print(f'L_tr:{L_tr}')
        loss = self.w_b*L_b + self.w_e*torch.sum(L_e) +self.w_r*torch.sum(L_tr)
        # global self.count_loss
        # global self.stepsize_print
        # global step_count_back
        # global t3
        # global t4
        self.trace_deviation = torch.mean(L_tr)
        # self.count_loss += 1
        return loss

    def train_epoch(self,optimizer,scheduler):
        self.L2 = 0.
        self.count_epoch += 1
        n = self.states.shape[0]//self.batch_size
        start = 0
        j = 0
        # print(n)
        for i in range(n+1):
            end = min(start+self.batch_size,self.states.shape[0])
            optimizer.zero_grad()
            if end==start:
                break
            # print(f'strat:{start}   end:{end}')
            loss = self.loss(self.states[start:end],
                          self.target_initial[start:end],
                          self.coef_initial[start:end])/\
                    (end-start)*self.states.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.L2 += loss

            start = end
            j += 1
        self.L2 = self.L2/j
        return self.L2
    
    def solve(self,upper_epochs=1000):
        self.training_loss = []
        if self.method=='Adam':
            gamma = 0.9
            # milestones = [5000,10000,20000,30000]
            optimizer = torch.optim.Adam(self.rho.parameters(),lr=1.e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                    step_size=3000, gamma=gamma, last_epoch=- 1)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            #             milestones=config.advmc_milestones,gamma=gamma)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            #               mode='min', factor=0.8,  patience = 2, threshold = 1e-5, 
            #               threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-06)
            self.count_epoch = 0
            self.t1 = time.time()
            while (self.count_epoch==0 or self.L2>1e-5) and self.count_epoch<upper_epochs :
                self.train_epoch(optimizer,scheduler)
                self.training_loss.append(self.L2)
                # print(f'epoch: {self.count_epoch}       loss:{self.L2}',
                    #   flush=True)
                if self.count_epoch%self.stepsize_print==0:
                    self.t2 = time.time()
                #     t2 = time.time()
                #     rho_0 = rho.rho_0().detach()
                #     trace = torch.real(torch.trace(rho_0))
                    print(f"epoch:{self.count_epoch:4d}   ll:{self.L2:.4e}  "   
                          f"t:{self.t2-self.t1:.2e}     "   
                            f"trace_deviation:{self.trace_deviation}    "     
                            f"lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
                            ,flush=True)
                    self.t1 = self.t2
                if self.count_epoch%self.stepsize_saverho==0:
                    torch.save(self.rho.state_dict(),f'rho_'+self.method+'ll_{self.count_epoch}')
        return self.L2
            # i = 0
            # L = self.loss()
            # rho_0 = self.rho.rho_0().detach()
            # trace = torch.real(torch.trace(rho_0))
            # print(f"count:{i:4d}   ll:{L:.4e}       trace:{trace}       lr:{optimizer.state_dict()['param_groups'][0]['lr']}",flush=True)
            # t1 = time.time()

            # if i%200==0:
            #     t2 = time.time()
            #     rho_0 = rho.rho_0().detach()
            #     trace = torch.real(torch.trace(rho_0))
            #     print(f"count:{i:4d}   ll:{L:.4e}   t:{t2-t1:.2e}    trace:{trace}      lr:{optimizer.state_dict()['param_groups'][0]['lr']}",flush=True)
            #     t1 = t2
            # if i%5000==0:
            #     torch.save(rho.state_dict(),f'rho_ll_{i}')
# class MyTakeStep:
#     def __init__(self,stepsize):
#         self.step = stepsize
#         self.rng = np.random.default_rng()
#     def __call__(self, x):
#         # print('average(para) %f' % np.average(x))

#         if np.abs(np.average(x)) > 10:
#             print('overflow,then restart\n')
#             x = self.rng.uniform(-4,0,x.shape)
#             # print('average(new para) %f' % np.average(x))
#         x_next = self.unirandom(x)
#         while(np.average(x_next) > 5):
#             print(f'average over 5:{np.average(x_next)}\n')
#             x_next = self.unirandom(x)
#         x = x_next
#         rho.vec_to_nn(torch.from_numpy(x))
#         global count
#         global meth
#         torch.save(rho.state_dict(),f'rho_simulate_ll_{meth}_{step_count+1}_0')
#         count = 0
#         return x
#     def unirandom(self,x) :
#         s = self.step
#         return x + self.rng.uniform(-s,s,x.shape)

# def print_fun(x,f,accepted):
#     #print the info of a sub-optimization of the basinhopping method
#     global step_count
#     global meth
#     step_count += 1
#     t_hop.append(time.time())
#     print("step:%2d, time_step = %.2f, accepted %d, loss at minimum %.5e" % (step_count,t_hop[-1]-t_hop[-2],int(accepted),f))
#     rho.vec_to_nn(torch.from_numpy(x))

#     rho_0 = rho.rho_0().detach()
#     n_up, n_down = operators.occupation(rho_0)
#     # print(f'{N_rho[328]:1d} {rho.States(statest[328:329]).item(): .6e}  {target[328]: .6e}')
#     # print(f'{N_rho[1270]:1d} {rho.States(statest[1270:1271]).item(): .6e}  {target[1270]: .6e}')
#     # print(f'{N_rho[666]:1d} {rho.States(statest[666:667]).item(): .6e}  {target[666]: .6e}')
#     # print(f'{N_rho[1460]:1d} {rho.States(statest[1460:1461]).item(): .6e}  {target[1460]: .6e}')
#     # print(f'{N_rho[1850]:1d} {rho.States(statest[1850:1851]).item(): .6e}  {target[1850]: .6e}')
#     # print(f'{N_rho[620]:1d} {rho.States(statest[620:621]).item(): .6e}  {target[620]: .6e}')
#     # print(f'up:{n_up:6e} ,down:{n_down:6e} ,trace: {torch.real(torch.trace(rho_0)):.5e}')

#     print('loss = %.5e\n' % (loss(x)))

#     if isinstance(f,float) :
#         if accepted==1 and np.isnan(f)==False :
#             rho.vec_to_nn(torch.from_numpy(x))
#             torch.save(rho.state_dict(),f'rho_simulate_{meth}_{step_count}')
#     print('\n')
#     return 0