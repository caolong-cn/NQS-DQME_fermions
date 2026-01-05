import torch
import torch.nn as nn
from torch.func import grad,functional_call
from torch import vmap

from ..utils import SubscriptTrans1, SubscriptTrans0_torch
from ..global_defs import get_device




def flat_para(para_tensor):
    stacked = {k: g.flatten(start_dim=1) for k, g in para_tensor.items()}  # 展平非批处理维度
    combined = torch.cat([g for g in stacked.values()], dim=1)
    return combined

def Complex1_s(a,dtype) :
    m_real = torch.Tensor(a).uniform_(-0.3,0.02).type(dtype)
    m_img = torch.Tensor(a).normal_(0,0.05).type(dtype)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sd(a,b,dtype) :
    m_real = torch.Tensor(a,b).uniform_(-0.3,0.02).type(dtype)
    m_img = torch.Tensor(a,b).normal_(0,0.1).type(dtype)
    m = m_real + (1.j)*m_img
    return m

def Complex1_h(a,dtype) :
    m_real = torch.Tensor(a).uniform_(-0.01,0.01).type(dtype)
    m_img = torch.Tensor(a).normal_(0,0.01).type(dtype)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sh(a,b,dtype) :
    m_real = torch.Tensor(a,b).uniform_(-0.01,0.01).type(dtype)
    m_img = torch.Tensor(a,b).normal_(0,0.01).type(dtype)
    m = m_real + (1.j)*m_img
    return m

def Complex2_dh(a,b,dtype) :
    m_real = torch.Tensor(a,b).uniform_(-0.01,0.01).type(dtype)
    m_img = torch.Tensor(a,b).normal_(0,0.01).type(dtype)
    m = m_real + (1.j)*m_img
    return m

def Complex2_sa(a,b,dtype) :
    m_real = torch.Tensor(a,b).uniform_(-0.01,0.01).type(dtype)
    m_img = torch.Tensor(a,b).normal_(0,0.01).type(dtype)
    m = m_real + (1.j)*m_img
    return m


count = 0
class NADO(nn.Module):
    def __init__(self,system_level,spin_degree,environment_model,bath_number,nsgn,
        device,table0
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

        self.count = 0

        self.float_dtype = torch.float64
        self.complex_dtype = torch.complex128
        self.grad_dtype = torch.float64

    def forward(self, x):
        return self.States(x)
    
    def States(self, x):    
        pass

    def State(self, x):    
        pass
    
    def compute_batched_grads(self, x_batch):
        params = dict(self.named_parameters())

        def forward_fn(params, x):
            return functional_call(self, params, x)

        def forward_fn_real(params,x):
            with torch.autocast(device_type=get_device().type,dtype=self.grad_dtype):
                output = forward_fn(params, x).real
            return output[0] 
        
        def forward_fn_img(params,x):
            with torch.autocast(device_type=get_device().type,dtype=self.grad_dtype):
                output = forward_fn(params, x).imag
            return output[0] 

        grad_fn_real = grad(forward_fn_real, argnums=0)
        batch_grads_real = flat_para(vmap(grad_fn_real, in_dims=(None, 0))(params, x_batch.unsqueeze(1))).detach()
        grad_fn_img = grad(forward_fn_img, argnums=0)
        batch_grads_img = flat_para(vmap(grad_fn_img, in_dims=(None, 0))(params, x_batch.unsqueeze(1))).detach()
        batch_grads = torch.zeros((len(x_batch),self.nparameters),dtype=torch.complex128,device=get_device())
        count = 0
        count_1 = 0
        for p in self.parameters():
            n_p = p.numel()
            if p.is_complex():
                batch_grads[:,count:count+n_p] = torch.real(batch_grads_real[:,count_1:count_1+n_p])+1.j*torch.real(batch_grads_img[:,count_1:count_1+n_p])
                batch_grads[:,count+n_p:count+2*n_p] = torch.imag(batch_grads_real[:,count_1:count_1+n_p])+1.j*torch.imag(batch_grads_img[:,count_1:count_1+n_p])
                count += 2*n_p
            else:
                batch_grads[:,count:count+n_p] =  batch_grads_real[:,count_1:count_1+n_p]+1.j*batch_grads_img[:,count_1:count_1+n_p]
                count += n_p
            count_1 += n_p
        return batch_grads  # output n×n_parameter matrix
    
    def f1(self, x1, y):
        return  x1.matmul(self.w_sh) + y.matmul(self.w_dh1) + self.bh

    def f2(self, x2 ,y):
        return torch.conj(x2.matmul(self.w_sh) + y.matmul(self.w_dh2)) + torch.conj(self.bh)

    def f3(self, x1, x2, y):
        a = (2*self.ba.type(self.complex_dtype) + 
            x1.matmul(self.w_sa) + torch.conj(x2.matmul(self.w_sa)) + 
            y.matmul((self.w_da + self.w_da).type(self.complex_dtype)))
        return a

    def sigmoid(self,x):  
        return 1/(1+torch.exp(-x))

    def n_para(self):
        n_para = 0
        for p in self.parameters():
            n = p.numel()
            if p.is_complex():
                n_para += 2*n
            else:
                n_para += n
        return n_para

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
        vec = torch.zeros(self.nparameters,dtype=self.float_dtype) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            i = j
            n = p.numel()
            j = i + n
            if p.is_complex():
                vec_complex = p.detach().flatten()
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
        gradient = torch.zeros(self.nparameters,dtype=self.float_dtype) 
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

    def rho_0(self) :
        #return rho0(the density operator)
        rho_0 = torch.zeros((2**self.Ns,2**self.Ns),dtype=self.complex_dtype,device=get_device())
        count_left = SubscriptTrans0_torch(self.table0[:,:self.Ns])
        count_right = SubscriptTrans0_torch(self.table0[:,self.Ns:])
        state0 = torch.zeros((self.table0.shape[0],self.nstate),dtype=torch.int8,device=get_device())
        state0[:,self.Nd:] = self.table0
        rho_0[count_left,count_right] = self.States(state0)
        return rho_0

    def rho(self) :
        #return all ADO
        m = 2**self.nstate
        rho = torch.zeros(m,dtype=self.complex_dtype)
        state = torch.zeros(self.nstate,dtype=torch.int8)
        for i in range(0,m):
            state = torch.Tensor(SubscriptTrans1(i,self.nstate))
            rho[i] = self.State(state)
        return rho

    def print_first(self):
        #print the first order ado
        n = 2**(self.Ns)
        N = self.Nd
        m = self.nstate
        rho_0 = torch.zeros((n,n),dtype=self.complex_dtype)
        state = torch.zeros(m,dtype=torch.int8)
        for k in range(0,self.Nd):
            state[k] = 1
            for i in range(0,n):
                state[N:N+self.Ns] = torch.Tensor(SubscriptTrans1(i,self.Ns))
                for j in range(0,n):
                    state[N+self.Ns:m] = torch.Tensor(SubscriptTrans1(j,self.Ns))
                    rho_0[i,j] = self.State(state)
            print('n:')
            print(state[0:N])
            print(rho_0)
            state[k] = 0
        return 0

    def rho_d(self) :
        #return all gradient
        m = 2**self.nstate
        rho_d = torch.zeros((m,self.torcharameters),dtype=self.complex_dtype)
        state = torch.zeros(self.nstate,dtype=torch.int8)
        for i in range(0,m):
            state = torch.Tensor(SubscriptTrans1(i,self.nstate))
            rho_d[i,:] = self.Gradient(state).reshape(1,-1)*self.State(state)
        return rho_d

    def print_gradient(self):
        gradient = torch.zeros(self.nparameters,dtype=self.float_dtype) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            print(name)
            print(p.grad)
        return gradient

    def print_parameters(self):
        gradient = torch.zeros(self.nparameters,dtype=self.float_dtype) 
        i = 0
        j = 0
        for name,p in self.named_parameters():
            print(name)
            print(p)
        return gradient
        
    def forward_2(self, x, lmbda=0.):    
        '''
        x1,x2:Ns; y:Nd
        x should be two-dimensional (*,self.nstate), * is the number of samples
        rho(x) should be nonzero
        '''
        y = x[:,0:self.Nd]
        N = torch.exp(lmbda*torch.sum(y,dim=1))
        rho_x = self.States(x)
        return torch.real(torch.multiply(rho_x,rho_x.conj()))/N

    def set_default_dtype(self,dtype):
        if dtype==torch.float64:
            self.float_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif dtype==torch.float32:
            self.float_dtype = torch.float32
            self.complex_dtype = torch.complex64
        return 0
    
    def set_grad_dtype(self,dtype):
        self.grad_dtype = dtype
        return 0
    

class RBM(NADO):
    def __init__(
    self,
    system_level,
    spin_degree,
    environment_model,
    bath_number,
    nsgn,
    hidden_number,
    ancillary_number,
    device,
    table0
    ):
        super().__init__(system_level,spin_degree,
                         environment_model,bath_number,nsgn,
                         device,table0)
        self.Nh = hidden_number
        self.Na = ancillary_number
        self.nparameters = 2*self.Ns+2*self.Nh+1*self.Nd+self.Na+2*(
            1*2*self.Nd*self.Ns+self.Ns*self.Nh+
                self.Ns*self.Na+2*self.Nh*self.Nd)+1*self.Na*self.Nd+0*self.Ns*self.Ns
        # 2*N_S+2*N_h+N_e+N_a+2*(2*N_e*N_s+Ns*Nh+Ns*Na+2*Nh*Ne)+Na*Ne
        # 2Nh + Na + 2(1 + Nh + Na)NS + (1 + 4Nh + Na)NE + 2NE NS 
        num_Nh = 2*(self.Nh+self.Ns*self.Nh+2*self.Nh*self.Nd)
        num_Na = self.Na+2*self.Ns*self.Na+self.Na*self.Nd

        self.bs = nn.Parameter(Complex1_s(self.Ns,self.complex_dtype))
        self.bd = nn.Parameter(torch.Tensor(self.Nd).type(self.float_dtype).uniform_(-1.8,-0.8))
        self.bsd0 = nn.Parameter(Complex2_sd(self.Ns, self.Nd,self.complex_dtype))
        self.bsd1 = nn.Parameter(Complex2_sd(self.Ns, self.Nd,self.complex_dtype))
        self.w_sh = nn.Parameter(Complex2_sh(self.Ns, self.Nh,self.complex_dtype))
        self.bh = nn.Parameter(Complex1_h(self.Nh,self.complex_dtype))
        self.w_sa = nn.Parameter(Complex2_sa(self.Ns, self.Na,self.complex_dtype))
        self.w_dh1 = nn.Parameter(Complex2_dh(self.Nd, self.Nh,self.complex_dtype))
        self.w_dh2 = nn.Parameter(Complex2_dh(self.Nd, self.Nh,self.complex_dtype))
        self.w_da = nn.Parameter(torch.Tensor(self.Nd, self.Na).type(self.float_dtype).uniform_(-0.01,0.01))
        self.ba = nn.Parameter(torch.Tensor(self.Na).type(self.float_dtype).uniform_(-0.01,0.01))

        self.nparameters = self.n_para()        
        print(self.nparameters)
        print(f'Nh:{num_Nh:d} ; Na:{num_Na:d} ; sd:{2*2*self.Nd*self.Ns:d}',flush=True)

    
    def States(self, x):   
        # self.count += x.shape[0]
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns],dim=1)
        coef = torch.floor(torch.sum(y[:,:self.Nd//2], dim=1)/2)+torch.floor(torch.sum(y[:,self.Nd//2:], dim=1)/2)
        y = y.type(self.complex_dtype)
        x1 = x1.type(self.complex_dtype)
        x2 = x2.type(self.complex_dtype)
        r1 = torch.exp(y.matmul((self.bd + self.bd).type(self.complex_dtype)) + 
            x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)) + 
            torch.sum(x1.matmul(self.bsd0).multiply(y) + x2.matmul(self.bsd1).multiply(y),dim=1))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y))),dim=1)
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y))),dim=1)
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y))),dim=1)
        r0 = r1 * r2 * r3_l * r3_r
        y_c = y.clone()
        y_c[:,0:self.Nbs] = y[:,self.Nbs:self.Nd]
        y_c[:,self.Nbs:self.Nd] = y[:,0:self.Nbs]
        x2_c = x1
        x1_c = x2
        r1_c = torch.exp(y_c.matmul((self.bd+self.bd).type(self.complex_dtype)) + 
                x1_c.matmul(self.bs) + torch.conj(x2_c.matmul(self.bs)) + 
                torch.sum(x1_c.matmul(self.bsd0).multiply(y_c) + x2_c.matmul(self.bsd1).multiply(y_c),dim=1))
        r2_c = torch.prod((1 + torch.exp(self.f3(x1_c, x2_c, y_c))),dim=1)
        r3_l_c = torch.prod((1 + torch.exp(self.f1(x1_c, y_c))),dim=1)
        r3_r_c = torch.prod((1 + torch.exp(self.f2(x2_c, y_c))),dim=1)
        r0_conj = torch.conj(r1_c * r2_c * r3_l_c * r3_r_c)
        rho_x = r0 + (-1)**(coef) * r0_conj
        return rho_x * 0.5**(2*self.Nh+self.Na)

    def State(self, x):   
        '''
        x1,x2:Ns; y:Nd
        x should be one-dimensional
        '''
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns])
        coef = torch.floor(torch.sum(y[:self.Nd//2])/2)+torch.floor(torch.sum(y[self.Nd//2:])/2)

        y = y.type(self.complex_dtype)
        x1 = x1.type(self.complex_dtype)
        x2 = x2.type(self.complex_dtype)

        r1 = torch.exp(y.matmul((self.bd + self.bd).type(self.complex_dtype)) + 
                    x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)) + 
                    x1.matmul((self.bsd0).matmul(y)) + x2.matmul((self.bsd1).matmul(y)))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y))))
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y))))
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y))))
        r0 = r1 * r2 * r3_l * r3_r
        y_c = y.clone()
        y_c[0:self.Nbs] = y[self.Nbs:self.Nd]
        y_c[self.Nbs:self.Nd] = y[0:self.Nbs]
        x2_c = x1
        x1_c = x2
        r1_c = torch.exp(y_c.matmul((self.bd+self.bd).type(self.complex_dtype)) + 
                    x1_c.matmul(self.bs) + torch.conj(x2_c.matmul(self.bs)) + 
                    x1_c.matmul(self.bsd0.matmul(y_c)) + x2_c.matmul(self.bsd1.matmul(y_c)))
        r2_c = torch.prod((1 + torch.exp(self.f3(x1_c, x2_c, y_c))))
        r3_l_c = torch.prod((1 + torch.exp(self.f1(x1_c, y_c))))
        r3_r_c = torch.prod((1 + torch.exp(self.f2(x2_c, y_c))))
        r0_conj = torch.conj(r1_c * r2_c * r3_l_c * r3_r_c)
        r = r0 + (-1)**coef * r0_conj

        return r * 0.5**(2*self.Nh+self.Na)
    
class RBM_nonds(NADO):
    def __init__(
    self,
    system_level,
    spin_degree,
    environment_model,
    bath_number,
    nsgn,
    hidden_number,
    ancillary_number,
    device,
    table0
    ):
        super().__init__(system_level,spin_degree,
                         environment_model,bath_number,nsgn,
                         device,table0)
        self.Nh = hidden_number
        self.Na = ancillary_number
        self.nparameters = 2*self.Ns+2*self.Nh+1*self.Nd+self.Na+2*(self.Ns*self.Nh+
                self.Ns*self.Na+2*self.Nh*self.Nd)+1*self.Na*self.Nd+0*self.Ns*self.Ns
        num_Nh = 2*(self.Nh+self.Ns*self.Nh+2*self.Nh*self.Nd)
        num_Na = self.Na+2*self.Ns*self.Na+self.Na*self.Nd

        self.bs = nn.Parameter(Complex1_s(self.Ns))
        self.bd = nn.Parameter(torch.Tensor(self.Nd).type(self.float_dtype).uniform_(-1.8,-0.8))
        self.w_sh = nn.Parameter(Complex2_sh(self.Ns, self.Nh))
        self.bh = nn.Parameter(Complex1_h(self.Nh))
        self.w_sa = nn.Parameter(Complex2_sa(self.Ns, self.Na))
        self.w_dh1 = nn.Parameter(Complex2_dh(self.Nd, self.Nh))
        self.w_dh2 = nn.Parameter(Complex2_dh(self.Nd, self.Nh))
        self.w_da = nn.Parameter(torch.Tensor(self.Nd, self.Na).type(self.float_dtype).uniform_(-0.01,0.01))
        self.ba = nn.Parameter(torch.Tensor(self.Na).type(self.float_dtype).uniform_(-0.01,0.01))

        self.nparameters = self.n_para()
        print(self.nparameters)
        print(f'Nh:{num_Nh:d} ; Na:{num_Na:d} ; sd:{2*2*self.Nd*self.Ns:d}',flush=True)


    def States(self, x):   
        self.count += x.shape[0]
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns],dim=1)
        coef = torch.floor(torch.sum(y[:,:self.Nd//2], dim=1)/2)+torch.floor(torch.sum(y[:,self.Nd//2:], dim=1)/2)
        y = y.type(self.complex_dtype)
        x1 = x1.type(self.complex_dtype)
        x2 = x2.type(self.complex_dtype)
        r1 = torch.exp(y.matmul((self.bd + self.bd).type(self.complex_dtype)) + 
            x1.matmul(self.bs) + torch.conj(x2.matmul(self.bs)))
        r2 = torch.prod((1 + torch.exp(self.f3(x1, x2, y))),dim=1)
        r3_l = torch.prod((1 + torch.exp(self.f1(x1, y))),dim=1)
        r3_r = torch.prod((1 + torch.exp(self.f2(x2, y))),dim=1)
        r0 = r1 * r2 * r3_l * r3_r
        y_c = y.clone()
        y_c[:,0:self.Nbs] = y[:,self.Nbs:self.Nd]
        y_c[:,self.Nbs:self.Nd] = y[:,0:self.Nbs]
        x2_c = x1
        x1_c = x2
        r1_c = torch.exp(y_c.matmul((self.bd+self.bd).type(self.complex_dtype)) + 
                x1_c.matmul(self.bs) + torch.conj(x2_c.matmul(self.bs)) )
        r2_c = torch.prod((1 + torch.exp(self.f3(x1_c, x2_c, y_c))),dim=1)
        r3_l_c = torch.prod((1 + torch.exp(self.f1(x1_c, y_c))),dim=1)
        r3_r_c = torch.prod((1 + torch.exp(self.f2(x2_c, y_c))),dim=1)
        r0_conj = torch.conj(r1_c * r2_c * r3_l_c * r3_r_c)
        rho_x = r0 + (-1)**(coef) * r0_conj
        return rho_x * 0.5**(2*self.Nh+self.Na)
    

class MLP(NADO):
    def __init__(self,system_level,spin_degree,environment_model,bath_number,nsgn,
        hidden_number,device,table0):
        super().__init__(system_level,spin_degree,environment_model,bath_number,nsgn,
                         device,table0)
        self.Nh = hidden_number

        self.nn = nn.Sequential(
            nn.Linear(self.nstate,self.Nh,dtype=self.complex_dtype),
            nn.Sigmoid(),
            nn.Linear(self.Nh,self.Nh,dtype=self.complex_dtype),
            nn.Sigmoid(),
            nn.Linear(self.Nh,1,dtype=self.complex_dtype),
            nn.Sigmoid()
        )
        
        self.nparameters = self.n_para()
        print(self.nparameters)


    def States(self, x):    
        self.count += x.shape[0]
        y,x1,x2 = torch.split(x,[self.Nd, self.Ns, self.Ns],dim=1)
        coef = torch.floor(torch.sum(y[:,:self.Nd//2], dim=1)/2)+torch.floor(torch.sum(y[:,self.Nd//2:], dim=1)/2)
        y = y.type(self.complex_dtype)
        x1 = x1.type(self.complex_dtype)
        x2 = x2.type(self.complex_dtype)

        r0 = self.nn(x)

        x_c = x.clone()
        b = x[:,self.Nbs:self.Nd]
        x_c[:,self.Nbs:self.Nd] = x[:,0:self.Nbs]
        x_c[:,0:self.Nbs] = b
        b = x[:,self.Nd+self.Ns:self.nstate]
        x_c[:,self.Nd+self.Ns:self.nstate] = x[:,self.Nd:self.Nd+self.Ns]
        x_c[:,self.Nd:self.Nd+self.Ns] = b
        r0_conj = self.nn(x_c)

        rho_x = r0 + (-1)**(coef) * r0_conj
        return rho_x 
    # * 0.5**(2*self.Nh+self.Na)