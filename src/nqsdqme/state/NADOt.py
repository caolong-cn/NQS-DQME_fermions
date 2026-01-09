import torch.nn as nn
import torch
from ..utils import SubscriptTrans1, SubscriptTrans0_torch

from ..global_defs import get_device
import time

f_1 = lambda t: t
f_2 = lambda t: 0.*t
f_3 = lambda t: 0.*t
# f_t = [f_1,f_2,f_3]

class NADOt(nn.Module):
    def __init__(
        self,
        system_level,
        spin_degree,
        environment_model,
        bath_number,
        nsgn,
        device,
        table0,
        N_t=3,
        f_t=[f_1,f_2,f_3],
        gamma=-0.
        ):
        super().__init__()
        self.Nv = system_level
        self.No = spin_degree
        self.M = environment_model
        self.Nb = bath_number
        self.nsgn = nsgn
        self.device = device
        self.table0 = torch.tensor(table0).to(self.device)

        self.Ns = self.No * self.Nv 
        self.Nd = nsgn * self.No * self.Nv * self.Nb * self.M 
        self.Nbs = self.No * self.Nv * self.Nb * self.M
        self.nstate = self.Ns*2 + self.Nd

        sparsity_rho0 = self.table0[:,0:self.Ns] - self.table0[:,self.Ns:]
        self.sparsity = torch.unique(sparsity_rho0,dim=0)

        self.N_t = N_t
        self.f_t = f_t
        self.gamma = gamma

        self.count = 0

    def States(self, x, t=None):  
        if t==None:
            # print('without t in rho.States')
            t = 0.
        x_t = torch.zeros((x.shape[0],self.nstate+self.N_t),dtype=torch.complex128,device=get_device())
        x_t[:,0:self.nstate] = x
        for i in range(self.N_t):
            x_t[:,self.nstate+i] = self.f_t[i](t)
        # x_t[:,self.nstate+1] = self.f_2(t)
        # x_t[:,self.nstate+2] = self.f_3(t)

        #x should be torch.float64  
        self.count += x.shape[0]
        # y,x1,x2,t = torch.split(x,[self.Nd, self.Ns, self.Ns,1],dim=1)
        coef = (torch.floor(torch.sum(x[:,:self.Nbs], dim=1)/2)+\
            torch.floor(torch.sum(x[:,self.Nbs:self.Nd], dim=1)/2)).view(-1,1)
        expgamma = torch.exp(self.gamma*torch.sum(x[:,:self.Nd],dim=1)).view(-1,1)
        # y = y.type(torch.complex128)
        # x1 = x1.type(torch.complex128)
        # x2 = x2.type(torch.complex128)
        r0 = self.forward(x_t)
        x_c = x_t.clone()
        x_c[:,0:self.Nbs] = x_t[:,self.Nbs:self.Nd]
        x_c[:,self.Nbs:self.Nd] = x_t[:,0:self.Nbs]
        x_c[:,self.Nd:self.Nd+self.Ns] = x_t[:,self.Nd+self.Ns:self.nstate]
        x_c[:,self.Nd+self.Ns:self.nstate] = x_t[:,self.Nd:self.Nd+self.Ns]
        r0_conj = self.forward(x_c)
        rho_x = (r0 + (-1)**(coef) * torch.conj(r0_conj))*expgamma
        return rho_x.flatten() 

    def forward(self, x, t):    
        pass


    def get_parameter_number(self):
        n = 0
        for p in self.parameters():
            if p.is_complex():
                n += 2*p.numel()
            else:
                n += p.numel() 
        # self.nparameters = n
        return n



    def f0(self,x):
        #judge whether rho(x) is zero. if rho(x) = 0, return  False
        #mc/local_pre都与此有关
        Ns = self.Ns
        Nd = self.Nd
        No = self.No
        N = torch.div(Nd, 2, rounding_mode='floor')
        n_minus = torch.sum(x[0:N].reshape(Ns,-1),dim=(1)).flatten()
        n_plus = torch.sum(x[N:Nd].reshape(Ns,-1),dim=(1)).flatten()
        m_0 = x[Nd:Nd+Ns]
        m_1 = x[Nd+Ns:Nd+2*Ns]
        #order of level and spin in m and n is inverse
        #order of level*spin in m and n is inverse
        n_minus = n_minus.flip(0).reshape(No,-1).T.flatten()
        n_plus = n_plus.flip(0).reshape(No,-1).T.flatten()
        for i in range(0,self.sparsity.shape[0]):
            if all(n_minus - n_plus + m_0 - m_1 == self.sparsity[i]):
                return True
        return False


    def nn_to_vec(self):
        vec = torch.zeros(self.nparameters,dtype=torch.float64) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                vec_complex = p.detach().flatten()
                # print(vec_complex.shape)
                # print(i)
                # print(j)
                # print(vec.shape)
                vec[i:j] = torch.real(vec_complex)
                i = j
                j = j + n
                vec[i:j] = torch.imag(vec_complex)
            else:
                vec[i:j] = p.detach().flatten()
        return vec

    def vec_to_nn(self,vec):
        #vec should in device
        i = 0
        j = 0
        for p in self.parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                a = vec[i:j].reshape(p.shape)
                i = j
                j = j + n
                b = vec[i:j].reshape(p.shape)
                p.data = a+1.0j*b
            else:
                p.data = vec[i:j].reshape(p.shape)
        return self
    
    def nnd_to_vec(self):
        gradient = torch.zeros(self.nparameters,dtype=torch.float64) 
        i = 0
        j = 0
        for name,p in self.named_parameters() :
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex() :
                gradient_complex = p.grad.flatten()
                gradient[i:j] = torch.real(gradient_complex)
                i = j
                j = j + n
                gradient[i:j] = torch.imag(gradient_complex)
            else:
                gradient[i:j] = p.grad.flatten()
        return gradient

    def vec_to_nnd(self,gradient):
        i = 0
        j = 0
        for name,p in self.named_parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                a = gradient[i:j].reshape(p.shape)
                i = j
                j = j + n
                b = gradient[i:j].reshape(p.shape)
                p.grad = a+1.0j*b
            else:
                p.grad = gradient[i:j].reshape(p.shape)
        return gradient


    def ifasy(self,x) :
        y = torch.zeros(x.shape[0],dtype=int,device=x.device)
        N = self.Nd//2
        M = self.Ns
        y[0:N] = x[N:N*2]
        y[N:N*2] = x[0:N]
        y[N*2:N*2+M] = x[N*2+M:N*2+M*2]
        y[N*2+M:N*2+M*2] = x[N*2:N*2+M]
        if all(x == y) :
            return True
        return False



    def rho_0(self,t=0.) :
        #return rho0(the density operator)
        rho_0 = torch.zeros((2**self.Ns,2**self.Ns),dtype=torch.complex128,device=self.device)
        count_left = SubscriptTrans0_torch(self.table0[:,:self.Ns])
        count_right = SubscriptTrans0_torch(self.table0[:,self.Ns:])
        state0 = torch.zeros((self.table0.shape[0],self.nstate),dtype=torch.float64,device=self.device)
        state0[:,self.Nd:] = self.table0
        # state0[:,-1] = t
        # print(count_left)
        # print(count_right)
        # print(self.States(state0))
        rho_0[count_left,count_right] = self.States(state0,t).flatten()
        return rho_0


    def print_gradient(self):
        gradient = torch.zeros(self.nparameters,dtype=torch.float64) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            print(name)
            print(p.grad)
        return gradient

    def print_parameters(self):
        gradient = torch.zeros(self.nparameters,dtype=torch.float64) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            print(name)
            print(p)
        return gradient

    def parameters_t_initial(self,n1=None,n2=None):
        if n1 is None:
            n1 = self.nstate
        if n2 is None:
            n2 = self.nstate + self.N_t
        dict = self.state_dict()
        dict['nn.0.weight'][:,n1:n2] = 0+0.j
        # dict['nn.0.weight'][:,0:-1].real = torch.randn((self.nhidden,self.nstate))+0.1
        # dict['nn.6.bias'].real = -0.3
        # dict['nn.6.bias'].imag = 0.15
        self.load_state_dict(dict)
        return 0


    def t_to_0(self,n1=None,n2=None):
        if n1 is None:
            n1 = self.nstate
        if n2 is None:
            n2 = self.nstate + self.N_t
        dict = self.state_dict()
        dict['nn.0.weight'][:,n1:n2] = 0. +0.j
        self.load_state_dict(dict)
        return 0

    def t_to_0_random(self,n1=None,n2=None):
        if n1 is None:
            n1 = self.nstate
        if n2 is None:
            n2 = self.nstate + self.N_t
        dict = self.state_dict()
        dict['nn.0.weight'][:,n1:n2] = (torch.randn_like(dict['nn.0.weight'][:,n1:n2]))*(1.+1.j)*0.001
        # dict['nn.0.weight'][:,-1]*0.0001
        self.load_state_dict(dict)
        return 0
    

class MLPt(NADOt):
    def __init__(
        self,
        system_level,
        spin_degree,
        environment_model,
        bath_number,
        nsgn,
        device,
        table0,
        N_t=3,
        f_t=[f_1,f_2,f_3],
        gamma=-2.,
        nhidden=30
        ):
        super().__init__(system_level,spin_degree,environment_model,
        bath_number,nsgn,device,table0,N_t,f_t,gamma)

        self.nhidden = nhidden
        self.Nh = nhidden
        # nhidden = 50
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.nstate+self.N_t,nhidden,dtype=torch.complex128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(nhidden,1,dtype=torch.complex128)
        )
        self.nparameters = self.get_parameter_number()

    def forward(self, x_t):
        return self.nn(x_t)


class MLPt_res(NADOt):
    def __init__(
        self,
        system_level,
        spin_degree,
        environment_model,
        bath_number,
        nsgn,
        device,
        table0,
        N_t=3,
        f_t=[f_1,f_2,f_3],
        gamma=-0.,
        nhidden=30
        ):
        super().__init__(system_level,spin_degree,environment_model,
        bath_number,nsgn,device,table0,N_t,f_t,gamma)

        self.nhidden = nhidden

        identity = 0.5*torch.eye(nhidden,dtype=torch.complex128)
        # nhidden = 50
        self.nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.nstate+3,nhidden,dtype=torch.complex128),
            # torch.nn.BatchNorm1d(nhidden,dtype=torch.complex128),
            torch.nn.Sigmoid()
            )
        self.nn2 = torch.nn.Sequential(
            torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
            # torch.nn.BatchNorm1d(nhidden,dtype=torch.complex128),
            torch.nn.Sigmoid())
        self.nn2[0].weight.data = self.nn2[0].weight - identity
        # self.nn3 = torch.nn.Sequential(
        #     torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
        #     # torch.nn.BatchNorm1d(nhidden,dtype=torch.complex128),
        #     torch.nn.Sigmoid())
        # self.nn3[0].weight.data = self.nn3[0].weight - identity
        # self.nn4 = torch.nn.Sequential(
        #     torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
        #     # torch.nn.BatchNorm1d(nhidden,dtype=torch.complex128),
        #     torch.nn.Sigmoid())
        # self.nn4[0].weight.data = self.nn4[0].weight - identity
        # self.nn5 = torch.nn.Sequential(
        #     torch.nn.Linear(nhidden,nhidden,dtype=torch.complex128),
        #     # torch.nn.BatchNorm1d(nhidden,dtype=torch.complex128),
        #     torch.nn.Sigmoid())
        # self.nn5[0].weight.data = self.nn5[0].weight - identity
        self.nn6 = torch.nn.Sequential(
            torch.nn.Linear(nhidden,1,dtype=torch.complex128),
            torch.nn.Sigmoid())
        self.nparameters = self.get_parameter_number()

    def forward(self,x):
        r1 = self.nn1(x)
        r2 = self.nn2(r1) +r1
        # r3 = self.nn3(r2) +r2
        # r4 = self.nn4(r3) +r3
        # r5 = self.nn5(r4) +r4
        r6 = self.nn6(r2)
        return r6

    def parameters_t_initial(self,n1=None,n2=None):
        if n1 is None:
            n1 = self.nstate
        if n2 is None:
            n2 = self.nstate + self.N_t
        dict = self.state_dict()
        dict['nn1.0.weight'][:,n1:n2] = 0+0.j
        # dict['nn.0.weight'][:,0:-1].real = torch.randn((self.nhidden,self.nstate))+0.1
        # dict['nn.6.bias'].real = -0.3
        # dict['nn.6.bias'].imag = 0.15
        self.load_state_dict(dict)
        return 0
    