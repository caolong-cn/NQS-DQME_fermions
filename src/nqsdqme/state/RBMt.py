import torch.nn as nn
import torch
from ..utils import SubscriptTrans1, SubscriptTrans0_torch

from ..global_defs import get_device
import time
from .NADOt import NADOt, f_t
from .NADO import (Complex1_h,Complex1_s,Complex2_dh,Complex2_sa,
                   Complex2_sd,Complex2_sh)


class RBMt(NADOt):
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
        f_t=f_t,
        gamma=-0.,
        nhidden=30,
        nauxiliary=30
        ):
        super().__init__(system_level,spin_degree,environment_model,
        bath_number,nsgn,device,table0,N_t,f_t,gamma)

        self.nhidden = nhidden
        self.Nh = nhidden
        self.nauxiliary = nauxiliary
        self.Na = nauxiliary

        self.bs = nn.Parameter(Complex1_s(self.Ns,self.complex_dtype))
        self.bd = nn.Parameter(torch.Tensor(self.Nd).type(self.float_dtype).uniform_(-1.8,-0.8))
        self.bsd0 = nn.Parameter(Complex2_sd(self.Ns, self.Nd,self.complex_dtype))
        self.bsd1 = nn.Parameter(Complex2_sd(self.Ns, self.Nd,self.complex_dtype))
        
        self.bh = nn.Parameter(Complex1_h(self.Nh,self.complex_dtype))
        self.ba = nn.Parameter(torch.Tensor(self.Na).type(self.float_dtype).uniform_(-0.01,0.01))

        self.bt = nn.Parameter(torch.zeros(self.N_t,dtype=self.float_dtype))
        
        self.w_sh = nn.Parameter(Complex2_sh(self.Ns, self.Nh,self.complex_dtype))
        self.w_sa = nn.Parameter(Complex2_sa(self.Ns, self.Na,self.complex_dtype))
        self.w_dh1 = nn.Parameter(Complex2_dh(self.Nd, self.Nh,self.complex_dtype))
        self.w_dh2 = nn.Parameter(Complex2_dh(self.Nd, self.Nh,self.complex_dtype))
        self.w_da = nn.Parameter(torch.Tensor(self.Nd, self.Na).type(self.float_dtype).uniform_(-0.01,0.01))
        
        self.w_ta = nn.Parameter(torch.zeros((self.N_t,self.Na),dtype=self.float_dtype))
        self.w_th = nn.Parameter(torch.zeros((self.N_t,self.Nh),dtype=self.complex_dtype))


        self.nparameters = self.get_parameter_number()
        print(f'number of parameters: {self.nparameters}')

    def forward(self, x_t):
        y,x1,x2,t = torch.split(x_t,[self.Nd, self.Ns, self.Ns,self.N_t],dim=1)
        y = y.type(self.complex_dtype)
        t = t.type(self.complex_dtype)
        x1 = x1.type(self.complex_dtype)
        x2 = x2.type(self.complex_dtype)
        r1 = torch.exp(t.matmul((self.bt + self.bt).type(self.complex_dtype))+
            y.matmul((self.bd + self.bd).type(self.complex_dtype)) + 
            x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)) + 
            torch.sum(x1.matmul(self.bsd0).multiply(y) + x2.matmul(self.bsd1).multiply(y),dim=1))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y,t))),dim=1)
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y,t))),dim=1)
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y,t))),dim=1)
        r0 = r1 * r2 * r3_l * r3_r
        return r0.view(-1,1)
    
    def f1(self, x1, y,t):
        return  x1.matmul(self.w_sh) + y.matmul(self.w_dh1)+t.matmul(self.w_th) + self.bh

    def f2(self, x2 ,y,t):
        return torch.conj(x2.matmul(self.w_sh) + y.matmul(self.w_dh2)+t.matmul(self.w_th))+ torch.conj(self.bh)

    def f3(self, x1, x2, y,t):
        a = (2*self.ba.type(self.complex_dtype) + 
            x1.matmul(self.w_sa) + torch.conj(x2.matmul(self.w_sa)) + 
            y.matmul((self.w_da + self.w_da).type(self.complex_dtype))+
            t.matmul((self.w_ta + self.w_ta).type(self.complex_dtype)))
        return a


    def parameters_t_initial(self,n1=None,n2=None):
        if n1 is None:
            n1 = 0
        if n2 is None:
            n2 = self.N_t
        dict = self.state_dict()
        dict['w_ta'][n1:n2,:] = 0.
        dict['w_th'][n1:n2,:] = 0.+0.j
        # dict['nn.0.weight'][:,0:-1].real = torch.randn((self.nhidden,self.nstate))+0.1
        # dict['nn.6.bias'].real = -0.3
        # dict['nn.6.bias'].imag = 0.15
        self.load_state_dict(dict)
        return 0