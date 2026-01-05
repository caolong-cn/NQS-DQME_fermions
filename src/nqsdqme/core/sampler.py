import numpy as np
import torch
from scipy.special import comb
from ..global_defs import get_device



def Jtov(k,env_num,No,Nv) :
    # Nd: nsgn,No,Nv,Nb,M  ;c.data: Nv,No
    # No is the index of spin
    k = k%(env_num*No*Nv)//env_num
    return No*(k%Nv)+k//Nv

def mton(k,env_num,No,Nv,Ns,Nbs) :
    
    return  Ns-1-Jtov(k,env_num,No,Nv)+Ns*(k//Nbs)
# (self.env*self.nspin*self.nvar)


def mton_opposite(k,env_num,No,Nv,Ns,Nbs) :
    # Nd: nsgn,No,Nv,Nb,M  ;c.data: Nv,No
    # No is the index of spin
    # k = k%(env_num*No*Nv)//env_num
    return  Ns-1-Jtov(k,env_num,No,Nv)+Ns*(1-(k//Nbs))

class Sampler():
    def __init__(self,basis,size=4096,
                 nonmccut=2,allcut=3,
                 lmbda=-3.):
        self.nonmccut = nonmccut
        self.allcut = allcut
        self.lmbda = lmbda
        self.N_samples = size

        self.Nd = basis.Nd
        self.Nbs =basis.Nbs
        self.Ns = basis.Ns
        self.env = basis.env
        self.nstate = basis.nstate
        self.nspin  = basis.nspin
        self.nl = basis.nl
        self.sys_mode = basis.sys_mode

        if self.nonmccut<self.allcut:
            n = basis.N_exact
            i = np.random.randint(0,n,(self.N_samples))
            self.samples = basis.states[i].copy()
            self.MC_init()
            self.cal_mccoe()
        else:
            self.samples = []
            self.mc_coefficient = 0.


    def flip(self):
        '''
         flip and accept according to np.exp(self.lmbda*(Msum1-Msum0))
         flip(s) = 1-s, delta_s = (1-2s)*if_flip
        '''
        m =  self.Nd
        count = 0
        k = 0
        row_ind = np.arange(self.N_samples)
        flags = np.zeros_like(self.samples)
        flagsm = np.zeros_like(self.samples)
        flagsn = np.zeros_like(self.samples)
        #lambda不能设置的很负
        u = np.random.rand((2))
        if u[0] < 1./self.Nd :
            #flip two m which are both 0 or 1
            n_flip = np.random.randint(0,self.Ns,(self.N_samples))
            flagsm[row_ind,n_flip+m] =  1
            flagsn[row_ind,m+self.Ns+n_flip] =  1
            flags = flagsm + flagsn 
            flag_nondiff = 1-np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
            ds = flag_nondiff*flags*(1-2*self.samples)
            s1 = self.samples + ds
        elif u[0] < 2./self.Nd:
            if u[1]<0.5:
                #flip n and the relative m, which are different from each other
                #nsgn*nspin*nvar*Nb*M to nsgn*nspin*nvar
                m_flip = np.random.randint(0,self.Nd,(self.N_samples))
                n_flip = mton(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
                # Jtov(m_flip,self.env)+self.Ns*(m_flip//(self.env*self.nspin*self.nvar))
                flagsm[row_ind,m_flip] =  1
                flagsn[row_ind,m+n_flip] =  1
                flags = flagsm + flagsn 
                flag_diff = np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.samples)
                s1 = self.samples + ds
            else :
                #flip n and the relative m, which are equal to each other
                #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
                m_flip = np.random.randint(0,self.Nd,(self.N_samples))
                n_flip = mton_opposite(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
                # Jtov(m_flip,self.env)+self.Ns*(1-(m_flip//(self.env*self.nspin*self.nvar)))
                # m_flip//self.env
                # n_flip = (1 - n_flip//self.Ns)*self.Ns+n_flip%self.Ns
                flagsm[row_ind,m_flip] =  1
                flagsn[row_ind,m+n_flip] =  1
                flags = flagsm + flagsn 
                flag_nondiff = 1-np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_nondiff*flags*(1-2*self.samples)
                s1 =self.samples + ds
        else:
            # s1 = self.samples
            # ds = np.zeros_like(self.samples)
            #flip two m, which are different from each other
            #nspin*nvar*Nb*M
            if u[1]<0.5:
            #m-
                n_flip = np.random.randint(0,self.Ns,(self.N_samples)) #determine which spin to flip
                m_flip = np.random.randint(0,self.env,(self.N_samples)) #determine which two modes to flip
                m_flip_1 = np.random.randint(0,self.env,(self.N_samples))
                flagsm[row_ind,n_flip*self.env+m_flip] =  1
                flagsn[row_ind,n_flip*self.env+m_flip_1] =  1
                flags = flagsm + flagsn 
                flag_diff = np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.samples)
                s1 = self.samples + ds
            else :
            #m+              
                n_flip = np.random.randint(0,self.Ns,(self.N_samples)) #determine which spin to flip
                m_flip = np.random.randint(0,self.env,(self.N_samples)) #determine which two modes to flip
                m_flip_1 = np.random.randint(0,self.env,(self.N_samples))
                flagsm[row_ind,self.Nbs+n_flip*self.env+m_flip] =  1
                flagsn[row_ind,self.Nbs+n_flip*self.env+m_flip_1] =  1  
                flags = flagsm + flagsn 
                flag_diff = np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.samples)
                s1 = self.samples + ds
        Msum1 = np.sum(s1[:,0:self.Nd],axis=1).reshape(-1,1)
        # print(Msum1.shape)
        Msum0 = np.sum(self.samples[:,0:self.Nd],axis=1).reshape(-1,1)
        accept_mc = np.less(np.random.rand(self.N_samples,1),np.exp(self.lmbda*(Msum1-Msum0)))
        accept_allcut = np.less_equal(Msum1,self.allcut)
        accept_nonmccut = np.greater(Msum1,self.nonmccut)
        accept = accept_mc*accept_nonmccut*accept_allcut
        self.samples = self.samples + \
            accept*ds
        return self.samples
    
    def flip_phi(self,rho):
        '''
        flip and accept according to |rho(s)|^2 
        '''
        m =  self.Nd
        count = 0
        k = 0
        row_ind = np.arange(self.N_samples)
        flags = np.zeros_like(self.samples)
        flagsm = np.zeros_like(self.samples)
        flagsn = np.zeros_like(self.samples)
        #lambda不能设置的很负
        u = np.random.rand((2))
        if u[0] < 1/self.env :
            #flip two m which are both 0 or 1
            n_flip = np.random.randint(0,self.Ns,(self.N_samples))
            flagsm[row_ind,n_flip+m] =  1
            flagsn[row_ind,m+self.Ns+n_flip] =  1
            flags = flagsm + flagsn 
            flag_nondiff = 1-np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
            ds = flag_nondiff*flags*(1-2*self.samples)
            s1 = self.samples + ds
        elif u[1]<0.5:
        #flip n and the relative m, which are different from each other
        #nsgn*nspin*nvar*Nb*M to nsgn*nvar*nspin
            m_flip = np.random.randint(0,self.Nd,(self.N_samples))
            n_flip = mton(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
            # self.Ns-1-Jtov(m_flip,self.env)+self.Ns*(m_flip//(self.env*self.nspin*self.nvar))
            flagsm[row_ind,m_flip] =  1
            flagsn[row_ind,m+n_flip] =  1
            flags = flagsm + flagsn 
            flag_diff = np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
            ds = flag_diff*flags*(1-2*self.samples)
            s1 = self.samples + ds
        else :
            #flip n and the relative m, which are equal to each other
            #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
            m_flip = np.random.randint(0,self.Nd,(self.N_samples))
            n_flip = mton_opposite(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
            # self.Ns-1-Jtov(m_flip,self.env)+self.Ns*(1-(m_flip//(self.env*self.nspin*self.nvar)))
            # m_flip//self.env
            # n_flip = (1 - n_flip//self.Ns)*self.Ns+n_flip%self.Ns
            flagsm[row_ind,m_flip] =  1
            flagsn[row_ind,m+n_flip] =  1
            flags = flagsm + flagsn 
            flag_nondiff = 1-np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
            ds = flag_nondiff*flags*(1-2*self.samples)
            s1 =self.samples + ds
        with torch.no_grad():
            a = rho.States(torch.tensor(self.samples,device=get_device())).to('cpu').numpy()
            rho2_1 = np.real(np.multiply(a,a.conj())).reshape(-1,1)
        Msum1 = np.sum(s1[:,0:self.Nd],axis=1).reshape(-1,1)

        Msum0 = np.sum(self.samples[:,0:self.Nd],axis=1).reshape(-1,1)
        accept_mc = np.less(np.random.rand(self.N_samples,1),rho2_1/self.rho2_mc)
        accept_allcut = np.less_equal(Msum1,self.allcut)
        accept = accept_mc*accept_allcut
        self.samples = self.samples + \
            accept*ds

        with torch.no_grad():
            a = rho.States(torch.tensor(self.samples,device=get_device())).to('cpu').numpy()
            self.rho2_mc = np.real(np.multiply(a,a.conj())).reshape(-1,1)
        return self.samples

    def MC_init(self):
        '''
        initialize the states of MC sampling to satisfy the rank of all states is larger than mccut
        '''
        m =  self.Nd
        count = 0
        k = 0
        row_ind = np.arange(self.N_samples)
        # flags = np.zeros_like(self.samples)
        #lambda不能设置的很负
        while(not(np.all(np.sum(self.samples[:,0:self.Nd],axis=1)>self.nonmccut))):
            # k = k+1
            flags = np.zeros_like(self.samples)
            flagsm = np.zeros_like(self.samples)
            flagsn = np.zeros_like(self.samples)
            u = np.random.rand((1))
            if u[0]<0.5:
        #flip n and the relative m, which are different from each other
        #nsgn*nspin*nvar*Nb*M to nsgn*nvar*nspin
                m_flip = np.random.randint(0,self.Nd,(self.N_samples))
                n_flip = mton(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
                # self.Ns-1-Jtov(m_flip,self.env)+self.Ns*(m_flip//(self.env*self.nspin*self.nvar))
                flagsm[row_ind,m_flip] =  1
                flagsn[row_ind,m+n_flip] =  1
                flags = flagsm + flagsn 
                flag_diff = np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.samples)
                s1 = self.samples + ds
            else :
                #flip n and the relative m, which are equal to each other
                #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
                m_flip = np.random.randint(0,self.Nd,(self.N_samples))
                n_flip = mton_opposite(m_flip,self.env,self.nspin,self.nl,self.Ns,self.Nbs)
                # self.Ns-1-Jtov(m_flip,self.env)+self.Ns*(1-(m_flip//(self.env*self.nspin*self.nvar)))
                # m_flip//self.env
                # n_flip = (1 - n_flip//self.Ns)*self.Ns+n_flip%self.Ns
                flagsm[row_ind,m_flip] =  1
                flagsn[row_ind,m+n_flip] =  1
                flags = flagsm + flagsn 
                flag_nondiff = 1-np.abs(np.sum((flagsm-flagsn)*self.samples,axis=1)).reshape(-1,1)
                ds = flag_nondiff*flags*(1-2*self.samples)
                s1 =self.samples + ds
            Msum1 = np.sum(s1[:,0:self.Nd],axis=1).reshape(-1,1)
            Msum0 = np.sum(self.samples[:,0:self.Nd],axis=1).reshape(-1,1)
            accept_allcut = np.less_equal(Msum1,self.allcut)
            accept_greater = np.greater(Msum1,Msum0)
            self.samples = self.samples + \
                accept_greater*accept_allcut*ds
        return self.samples

    def cal_mccoe(self):
        '''
        calculate the sum of p(s) over all non-zero states
        '''
        self.mc_coefficient = 0.
        a = np.exp(self.lmbda)
        Ns = self.Ns
        Ne = self.env
        if self.sys_mode == 0 :
            if self.allcut > 4:
                print('allcut is to big.')
            N_ado = np.zeros((5))
            N_ado[0] = self.Nd*2**(Ns-1)
            N_ado[1] = Ne**2*(Ns+comb(Ns,2))*2**Ns
            N_ado[2] = Ne**3*(Ns*(Ns-1)+comb(Ns,3))*2**Ns + Ne*comb(Ne,2)*Ns*2**Ns
            N_ado[3] = 2**(Ns-1)*Ne**4*(2*comb(Ns,4)+Ns**2*(Ns-1)+Ns/2)-2**(Ns-1)*Ne**3*Ns**2+2**(Ns-1)*Ne**2*Ns/2
            # (2+2*Ns*Ne+(Ns*Ns+Ns/2)*Ne**2+(Ns*(Ns**2-1)/3)*Ne**3+
                        # (comb(Ns,4)*2+Ns**2*(Ns-1)+Ns/2)*Ne**4)*2**(Ns-1)
            # N_ado[4] = 2**Ns*comb(Ns,5)*Ne**5+pow(2,Ns-3)*(comb(Ns,4)*32*Ne**5+comb(Ns,3)*24*Ne**3*comb(Ne,2))+\
            #             2**(Ns-1)*(comb(Ns,3)*6*Ne**5+comb(Ns,2)*4*Ne**3*comb(Ne,2)+comb(Ns,2)*4*Ne*comb(Ne,2)**2
            #                        +Ns*2*comb(Ne,2)*comb(Ne,3))
        elif self.sys_mode == 1 :
            # N_ado = np.array([320.,3504.,26624.])
            N_ado = np.zeros((3))
            nstate = self.nstate
            statescols = [i for i in range(3,3+nstate)]
            states_Nd = np.sum(np.loadtxt('table_cut4.data',usecols=(statescols),dtype=np.int8)[:,0:self.Nd],axis=1).flatten()
            for i in range(N_ado.size):
                condition = states_Nd[:]==(i+1)
                N_ado[i] = states_Nd[condition].shape[0]
        print(f'N_ado:{N_ado}')
        for i in range(self.nonmccut+1,self.allcut+1):
            print(i)
            self.mc_coefficient += N_ado[i-1]*a**i 
        print(f'mcweight:{self.mc_coefficient}')
        return self.mc_coefficient

