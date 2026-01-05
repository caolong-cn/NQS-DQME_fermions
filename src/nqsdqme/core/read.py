import torch
import numpy as np
from ..utils import SubscriptTrans1
#Hsys, c, gamma, eta, num_energy
#nvar2,nspin,ncor,nalf,nsgn
#Nv, No, M, Nb, nsgn
#nof energy levels of system; spin; nof dissipatve syss; nof baths; nof sigma

def ReadHamilton() :
    hbar = 0.658211928
# read the dimension of relative parameters   
    nsgn = np.loadtxt("res_corr.data", max_rows=1, dtype = int) 
    nspin = np.loadtxt("res_corr.data", skiprows=1, max_rows=1, dtype = int)
    nvar2 = np.loadtxt("res_corr.data", skiprows=2, max_rows=1, dtype = int)
    nalf = np.loadtxt("res_corr.data", skiprows=3, max_rows=1, dtype = int)
    ncor = np.loadtxt("res_corr.data", skiprows=4, max_rows=1, dtype = int)

# read the parameters of Hs
    sys = np.loadtxt("input_h", max_rows=1, dtype = int)
    if sys==0:
        energy_level = np.loadtxt("input_h", skiprows=1, max_rows=2, dtype = np.float64).T.flatten()/hbar
        U = np.loadtxt("input_h", skiprows=3, max_rows=4, dtype = np.float64)/hbar
    elif sys==1 :
        energy_level = np.loadtxt("input_h", skiprows=1, max_rows=4, dtype = np.float64).T.flatten()/hbar
        U = np.loadtxt("input_h", skiprows=5, max_rows=1, dtype = np.float64)/hbar
        J = np.loadtxt("input_h", skiprows=6, max_rows=1, dtype = np.float64)/hbar
    else:
        print(f"we don't support this type of sys: {sys}")
# read cgama(nsgn,nspin,nvar2,ncor,nalf)
    cgama_real = np.loadtxt("res_corr.data", skiprows=5, usecols=10, dtype = np.float64)
    cgama_img = np.loadtxt("res_corr.data", skiprows=5, usecols=11, dtype = np.float64)
    cgama = np.zeros((cgama_real.shape), dtype=np.complex128)
# for i in range(cd_real.size):
#   cgama[i] = complex(cgama_real[i], cgama_img[i])
    cgama = cgama_real + (1.j)*cgama_img
    gamma = cgama
# gamma = np.stack(cgama[0::2],cgama[1::2])


# read cb as (nspin,nvar2,ncor,nalf,nsgn)
    cb_real = np.loadtxt("res_corr.data", skiprows=5, usecols=6, dtype = np.float64)
    cb_img = np.loadtxt("res_corr.data", skiprows=5, usecols=7, dtype = np.float64)
    cb = np.zeros((cb_real.shape), dtype=np.complex128)
# for i in range(cb_real.size):
#   cb[i] = complex(cb_real[i], cb_img[i])
    cb = cb_real + (1.j)*cb_img
    eta = np.transpose(np.reshape(cb,(nsgn,-1)))


# read ifff(nvar2,nspin,ncor,nalf,nsgn)
#ifff = np.loadtxt("res_corr.data", skiprows=5, usecols=12, dtype = int)
    if sys==0:
        return (sys,energy_level,U,gamma,eta,nvar2,nspin,ncor,nalf,nsgn)
    else :
        return (sys,energy_level,U,gamma,eta,nvar2,nspin,ncor,nalf,nsgn,J)
    # (torch.tensor(Hsys).type(torch.float64),torch.tensor(c).type(torch.float64),
    #     torch.tensor(gamma),torch.tensor(eta),torch.tensor(nvar2),
    #     torch.tensor(nspin),torch.tensor(ncor),torch.tensor(nalf),
    #     torch.tensor(nsgn))


def Readrho(Nd,M,file0='table0.data',file1='table1.data'):
    table_f0 = np.loadtxt(file1,dtype=int)
    rho_f0_real = np.loadtxt(file0,usecols=(0),dtype=np.complex128)
    rho_f0_img = np.loadtxt(file0,usecols=(1),dtype=np.complex128)
    rho_f0 = rho_f0_real + 1.j * rho_f0_img
    n = table_f0[0,:].size
    N = 2**n
    m = table_f0.shape[0]
    x = np.zeros(n,dtype=int)
    y = np.zeros(n,dtype=int)
    rho = np.zeros(N,dtype=np.complex128)

    for i in range(0,N):
        x = SubscriptTrans1(i,n)
        # N = rho.Nd//2
        # M = rho.Ns
        y[0:Nd] = x[Nd:Nd*2]
        y[Nd:Nd*2] = x[0:Nd]
        y[Nd*2:Nd*2+M] = x[Nd*2+M:Nd*2+M*2]
        y[Nd*2+M:Nd*2+M*2] = x[Nd*2:Nd*2+M]
        for j in range(0,m):
            if all(table_f0[j,:] == x):
                rho[i] = rho_f0[j]
                break
            elif all(table_f0[j,:] == y):
                rho[i] = np.conj(rho_f0[j])
                break
    return torch.tensor(rho)