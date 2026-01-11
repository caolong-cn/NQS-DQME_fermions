import torch
import numpy as np
import sys
import os
import time
import scipy.optimize as spop
from scipy.special import comb


from ..utils import SubscriptTrans0_torch
from ..global_defs import get_device


class BasicSPOP():
    def __init__(self,rho,states:torch.Tensor,operators):
        self.rho = rho
        self.states = states
        self.operators = operators
        self.N_rho = torch.sum(states[:,:self.rho.Nd],axis=1).to(torch.int64)



    def loss(self,vec_of_rho):
        pass

    def loss_d(self,vec_of_rho) :
        pass

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
        n_up, n_down = self.operators.occupation(rho_0)
        print(f'{self.N_rho[328]:1d} {self.rho.States(self.states[328:329]).item(): .6e}  {self.target[328]: .6e}')
        print(f'{self.N_rho[1270]:1d} {self.rho.States(self.states[1270:1271]).item(): .6e}  {self.target[1270]: .6e}')
        print(f'{self.N_rho[666]:1d} {self.rho.States(self.states[666:667]).item(): .6e}  {self.target[666]: .6e}')
        print(f'{self.N_rho[1460]:1d} {self.rho.States(self.states[1460:1461]).item(): .6e}  {self.target[1460]: .6e}')
        print(f'{self.N_rho[1850]:1d} {self.rho.States(self.states[1850:1851]).item(): .6e}  {self.target[1850]: .6e}')
        print(f'{self.N_rho[620]:1d} {self.rho.States(self.states[620:621]).item(): .6e}  {self.target[620]: .6e}')
        print(f'up:{n_up:6e} ,down:{n_down:6e} ,trace: {torch.real(torch.trace(rho_0)):.5e}')

        print('loss = %.5e\n' % (self.loss(x)))
        torch.save(self.rho.state_dict(),f'rho_simulate_{self.method}_{self.count_hopping}')
        print('\n')
        return 0
    
    def hopping(self,meth='TNC',niter=5,T=0.001,step_h=0.65,tol=1e-10,maxfun=10000,step_p=1000,step_s=5000):
        #tnc求解全局最小值
        #主要由'maxfun'：Max. number of function evaluations来调控每一次优化，一般设置为10000-15000即可
        #可优化至约1e-2量级
        #tnc优化必须是float64
        mytakestep = self.MyTakeStep(step_h)
        self.method = meth
        self.step_p = step_p
        self.step_s = step_s
        result = spop.basinhopping(self.loss,self.rho.nn_to_vec(),niter=niter,T=T,
            take_step=mytakestep,callback=self.print_fun,
            minimizer_kwargs={'method':self.method,'jac':self.loss_d,'tol':tol,'options':{'maxfun':maxfun}})
        self.result_print(result)
        return result

    def optimization(self,meth='BFGS',tol=1e-8,step_p=50,step_s=200):
        self.method = meth
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
        self.result_print(result)
        torch.save(self.rho.state_dict(),'rho_BFGS_'+str(self.rho.Nh))

        rho_0 = self.rho.rho_0().detach()
        print(f'origin_trace: {torch.real(torch.trace(rho_0)): .3e}\n',flush=True)
        print(f'rho_0: {rho_0}')
        return 0 
      
    def result_print(self,result):
        '''
        print the results of the total basinhopping
        result is the returned value of spop.basinhopping
        '''
        print("n_it: %d" % result.nit)
        print("total evaluations: %d" % result.nfev,flush=True)
        print(f'terminate because:{result.message}')
        self.rho.vec_to_nn(torch.from_numpy(result.x).to(get_device()))

        rho_0 = self.rho.rho_0().detach()
        n_up, n_down = self.operators.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))
        print('rho_0:')
        print(rho_0)
        print(f'up:{n_up:6e}  ;down:{n_down:6e}')
        print(f'trace: {trace:.3e}')
        print('loss_at_end')
        print(f'loss:{self.loss(result.x):.5e}')
        return 0

class SS_pre(BasicSPOP):
    def __init__(self,allcut,rho,states:torch.Tensor,operators):
        super().__init__(rho,states,operators)
        rho_input = torch.from_numpy(np.loadtxt("table_cut"+str(allcut+1)+".data",usecols=(0,1)))
        self.target = (rho_input[:,0] + 1.j*rho_input[:,1]).to(get_device())
        self.coef = 3**torch.sum(states[:,:self.rho.Nd],dim=1,dtype=torch.float64)

        self.method = 'TNC'#'BFGS'
        self.count = 0
        self.step_p = 100
        self.step_s = 500
        self.count_hopping = 0
        self.t1 = time.time()
        self.t2 = time.time()


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
            self.t2 = time.time()
            print(f'count:{self.count:4d}   loss_{self.method}:{L:.4e}   t:{self.t2-self.t1:.2e}     trace:{trace}',flush=True)
            self.t1 = time.time()
        # if self.count%min(10,self.count_0)==0:
        #     self.result_print(L,ifresult=False)
        if self.count!=0 and self.count%self.step_s==0:
            torch.save(self.rho.state_dict(),f'rho_{self.method}_{self.count}')
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
        n_up, n_down = self.operators.occupation(rho_0)
        trace = torch.real(torch.trace(rho_0))
        I = self.operators.current_general(self.rho,rho_0)
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
            S12, Sx2, Sy2, Sz2 = self.operators.spin_ddot(rho_0)
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
            self.t2 = time.time()
            print(f'count:{self.count:4d}   ll_{self.method}:{L:.4e}   t:{self.t2-self.t1:.2e}     trace:{trace}',flush=True)
            self.t1 = time.time()
        # if self.count%min(10,self.count_0)==0:
        #     self.result_print(L,ifresult=False)
        if self.count!=0 and self.count%self.step_s==0:
            torch.save(self.rho.state_dict(),f'rho_{self.method}_{self.count}')
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
        self.method = meth
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
        self.result_print(result)
        torch.save(self.rho.state_dict(),'rho_BFGS_'+str(self.rho.nhidden))

        rho_0 = self.rho.rho_0().detach()
        print(f'origin_trace: {torch.real(torch.trace(rho_0)): .3e}\n',flush=True)
        print(f'rho_0: {rho_0}')
        return 0