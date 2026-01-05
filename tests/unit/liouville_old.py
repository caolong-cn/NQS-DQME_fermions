import numpy as np
import scipy.sparse as sp
from numba import jit
import time
import sys
import torch
from scipy.special import comb

import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from nqsdqme.core.liouville import Liouville



class LiouvilleTest(Liouville):
    def __init__(self,read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.) :
        '''
        read : the information of Hs
        nonmccut : trancation of direct summation part
        allcut : trancation
        notsaving = 0(1): (not) saving L
        '''
        super.__init__(read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.)

    def test_L(self):
        self.L={}
        self.L['case1_hight_st_cut2'] = 7.77015158e-05
        self.L['case1_hight_t_cut2'] = 2.01111306e-02
        self.L['case1_hight_t_cut3'] = 2.0079092237420593e-2
        self.L['case1_lowt_t_cut3'] = None
        self.L['case2_hight_st_cut2'] = 4.217876191760257e-4
        self.L['case2_hight_t_cut2'] = 1.710977e-02 
        self.L['case2_hight_t_cut3'] = 1.680703e-02
        self.L['case2_lowt_t_cut3'] = 2.123621e-02
        return self.L 

    def calc_is(self,iorb,s,norb,nspin):
        '''
        iorb: ith orb(from 0 to n-1)
        s: ith spin(for nspin=2, s=0=up, s=1=down)
        norb,nspin the total number of orbitals and spins
        the order of orbitals and spins: n-down,n-up;....;1-down,1-up
        '''
        n = norb*nspin
        c_is = np.zeros((2**n),dtype=torch.int32)
        i = iorb*2+s
        len0 = 2**(i+1)
        nbef = (n-1) - i
        naft = i
        for i_bef in range(2**nbef):
            'the number before i'
            nrow = i_bef*len0
            ncolumn = nrow + 2**i
            for i_aft in range(2**naft):
                'the number after i'
                c_is[nrow+i_aft,ncolumn+i_aft] = 1
        return c_is

class LiouvilleNonbatched(Liouville):
    def __init__(self,read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.) :
        '''
        read : the information of Hs
        nonmccut : trancation of direct summation part
        allcut : trancation
        notsaving = 0(1): (not) saving L
        '''
        super.__init__(read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.)

    def H_ls_d(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s> and the states s
        suitable to states(0,1,3,2,4)
        '''
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        No = rho.No
        Nv = rho.Nv
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((1,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # count += 1
        # #states_need[count] = state
        if self.sys==0:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                  +self.gamma_tr.dot(state[0:N])
            count += 1
        elif self.sys==1:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                 +self.gamma_tr.dot(state[0:N])\
                   +(-1.0j)*(-0.25*self.J)*((m_0[0]-m_0[1])*(m_0[2]-m_0[3])-(m_1[0]-m_1[1])*(m_1[2]-m_1[3]))
            count += 1
        else :
            print(f"we don't support this type of system: {self.sys}")
            count += 1
        return count, L_pre[0:count], states_need[0:count]

    def H_ls_nd(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        suitable to states(0,1,3,2,4)
        '''
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        No = rho.No
        Nv = rho.Nv
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # count += 1
        # #states_need[count] = state
        if self.sys==1:
            if np.sum(m_0[0:2])==1 and np.sum(m_0[2:])==1 :
                A = 0.
                if m_0[0]==0:
                    states_need[count,N] = 1
                    states_need[count,N+1] = 0
                    A = 1.j
                else :
                    states_need[count,N] = 0
                    states_need[count,N+1] = 1
                    A = -1.j
                if m_0[2]==0:
                    states_need[count,N+2] = 1
                    states_need[count,N+3] = 0
                    A *= 1.j
                else :
                    states_need[count,N+2] = 0
                    states_need[count,N+3] = 1
                    A *= -1.j
                L_pre[count] = (-1.0j)*(-0.25*self.J*(1+A))
                count += 1
            if np.sum(m_1[0:2])==1 and np.sum(m_1[2:])==1 :
                A = 0.
                if m_1[0]==0:
                    states_need[count,N+rho.Ns] = 1
                    states_need[count,N+rho.Ns+1] = 0
                    A = -1.j
                else :
                    states_need[count,N+rho.Ns] = 0
                    states_need[count,N+rho.Ns+1] = 1
                    A = 1.j
                if m_1[2]==0:
                    states_need[count,N+rho.Ns+2] = 1
                    states_need[count,N+rho.Ns+3] = 0
                    A *= -1.j
                else :
                    states_need[count,N+rho.Ns+2] = 0
                    states_need[count,N+rho.Ns+3] = 1
                    A *= 1.j
                L_pre[count] = (1.0j)*(-0.25*self.J*(1+A))
                count += 1
        else :
            print(f"we don't support this type of system: {self.sys}")
            count += 1
        return count, L_pre[0:count], states_need[0:count]
    
    
    def H_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        suitable to states(0,1,3,2,4)
        '''
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        No = rho.No
        Nv = rho.Nv
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # count += 1
        # #states_need[count] = state
        if self.sys==0:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                  +self.gamma_tr.dot(state[0:N])
            count += 1
        elif self.sys==1:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                 +self.gamma_tr.dot(state[0:N])\
                   +(-1.0j)*(-0.25*self.J)*((m_0[0]-m_0[1])*(m_0[2]-m_0[3])-(m_1[0]-m_1[1])*(m_1[2]-m_1[3]))
            count += 1
            if np.sum(m_0[0:2])==1 and np.sum(m_0[2:])==1:
                A = 0.
                if m_0[0]==0:
                    states_need[count,N] = 1
                    states_need[count,N+1] = 0
                    A = 1.j
                else :
                    states_need[count,N] = 0
                    states_need[count,N+1] = 1
                    A = -1.j
                if m_0[2]==0:
                    states_need[count,N+2] = 1
                    states_need[count,N+3] = 0
                    A *= 1.j
                else :
                    states_need[count,N+2] = 0
                    states_need[count,N+3] = 1
                    A *= -1.j
                L_pre[count] = (-1.0j)*(-0.25*self.J*(1+A))
                count += 1
            if np.sum(m_1[0:2])==1 and np.sum(m_1[2:])==1:
                A = 0.
                if m_1[0]==0:
                    states_need[count,N+rho.Ns] = 1
                    states_need[count,N+rho.Ns+1] = 0
                    A = -1.j
                else :
                    states_need[count,N+rho.Ns] = 0
                    states_need[count,N+rho.Ns+1] = 1
                    A = 1.j
                if m_1[2]==0:
                    states_need[count,N+rho.Ns+2] = 1
                    states_need[count,N+rho.Ns+3] = 0
                    A *= -1.j
                else :
                    states_need[count,N+rho.Ns+2] = 0
                    states_need[count,N+rho.Ns+3] = 1
                    A *= 1.j
                L_pre[count] = (1.0j)*(-0.25*self.J*(1+A))
                count += 1
        else :
            print(f"we don't support this type of system: {self.sys}")
            count += 1
        return count, L_pre[0:count], states_need[0:count]
        
    def L_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        suitable to states(0,1,3,2,4)
        '''
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        No = rho.No
        Nv = rho.Nv
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # count += 1
        # #states_need[count] = state
        count_, L_pre_, states_need_ = self.H_ls(rho,state)
        count += count_
        L_pre[0:count_] = L_pre_
        states_need[0:count_] = states_need_
      
        i_min,i_min_1 = Nminus,Nminus
        i_pos,i_pos_1 = Npositive,Npositive
        l_0 = 0
        l_1 = 0
        i_0,i_0_1 = 0,0
        i_1,i_1_1 = 0,0   
        #consider nminus/npositive as a Ns*(Nb*M) matrix and calculate L_ls row-wisely
        #i_min/i_pos: the sum of elements in nminus/npos whose indices are larger than i
        #i_0/i_1: the sum of elements in nminus/npos whose indices are smaller than i
        #l_0/l_1: the sum of elements in m_0/m_1 whose indices are smaller than i
        for v in range(0,M):
            if v==0:
                if self.sys==0:
                    l_0 = m_0[1]
                    l_1 = m_1[1]
                else :
                    l_0 = np.sum(m_0[2:]) #no flip
                    l_1 = np.sum(m_1[2:])
            elif v==1:
                if self.sys==0:
                    l_0 = 0.
                    l_1 = 0.
                else:
                    l_0 = np.sum(m_0[2:]) +m_0[0]
                    l_1 = np.sum(m_1[2:]) +m_1[0]
            elif v==2:
                l_0 = m_0[3]
                l_1 = m_1[3]
            else :
                l_0 = 0.
                l_1 = 0.
            
            # l_0 = l_0 + m_0[v] 
            w = M-v-1 #no_flip
            v_tr = Nv*(w%No)+w//No
            if m_0[v] == 0 :
                # print(f'v:{v}')
                # print(f'w:{w}')
                # print(f'v_tr:{v_tr}') 
                for m in range(0,env_num):
                # #5th
                    k = v_tr*env_num+m
                    # i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        i_0_1 = np.sum(nminus[0:k])
                        states_need[count,k] = 0
                        states_need[count,N+v] = 1
                        L_pre[count] = (-1.j)*pow(-1,l_0+i_0_1+Nminus-1)*self.eta_tr[0,k]
                        # i_0_1+l_0+
                        # (-1.j*pow(-1,i_0_1+l_0+Nminus)*self.eta_tr[0,k]
                        count += 1
                #3th
                    # i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        i_1_1 = np.sum(npositive[0:k])
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+v] = 1           
                            L_pre[count] = (-1.0j)*pow(-1,l_0+i_1_1+Nminus+Npositive)
                            # l_0+i_1_1+
                            # (-1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive)
                            count += 1
            #     i_0_1 = i_0
            #     i_1_1 = i_1
            if m_0[v]==1 :
                for m in range(0,env_num):
                #1th
                    k = v_tr*env_num+m
                    # i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        i_min_1 = np.sum(nminus[k:])
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+v] = 0
                            L_pre[count] = (-1.0j)*pow(-1,i_min_1+l_0)
                            count += 1
        # #         #7th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        i_pos_1 = np.sum(npositive[k+1:])
                        states_need[count,N_half+k] = 0
                        states_need[count,N+v] = 0
                        L_pre[count] = (-1.j)*pow(-1,i_pos_1+l_0+Nminus)*self.eta_tr[1,k]
                        # (-1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k]
                        count += 1
        #         i_min_1 = i_min
        #         i_pos_1 = i_pos
        #     l_1 = l_1 + m_1[v]
            if m_1[v]==0 :
                for m in range(0,env_num):
                    k = v_tr*env_num+m
                    #2th
                    # i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        i_min_1 = np.sum(nminus[k+1:])
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+rho.Ns+v] = 1
                            L_pre[count] = -(-1.0j)*pow(-1,i_min_1+l_1+Nminus+Npositive)
                            # 
                            count += 1
                    #8th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        i_pos_1 = np.sum(npositive[k+1:])
                        states_need[count,N_half+k] = 0
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] =-(-1.j)*pow(-1,i_pos_1+l_1+Npositive-1)*(self.eta_tr[0,k]).conj()
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos       
            if m_1[v]==1 :
                for m in range(0,env_num):
                    k = v_tr*env_num+m
        # #             #6th
                    # i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        i_0_1 = np.sum(nminus[0:k])
                        states_need[count,k] = 0
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = -(-1.j)*pow(-1,i_0_1+l_1+Npositive)*(self.eta_tr[1,k]).conj()
                        count += 1
        # #             #4th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        i_1_1 = np.sum(npositive[0:k])
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+rho.Ns+v] = 0
                            L_pre[count] = -(-1.0j)*pow(-1,i_1_1+l_1)
                            count += 1

        return count, L_pre[0:count], states_need[0:count]
    
    def Hforward_mc(self,rho,N = 5000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        self.flip()
        states_mc = self.states_mc
        # print(states_mc.shape)
        # print(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1))
        weight = torch.tensor(np.exp(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1)),device=self.device)
        # .reshape(-1,1)
        n = self.mc_size
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.H_ls(rho,states_mc[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                rho_states = rho.States(states_need_torch)
                L_forward_pre = torch.multiply(rho_states,L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward, weight, n
    
    def Lforward_old(self,rho) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        the L_rho is calculated in this function
        _old means for bacnhmark with old method
        '''
        n = self.states.shape[0]
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls_old(rho,self.states[i])
            L_pre_0 = torch.tensor(L_pre_0,device=self.device)
            states_need_0 = torch.tensor(states_need_0,device=self.device)
            L_forward[i] = torch.dot(rho.States(states_need_0),(L_pre_0))
        return L_forward

    def Llocals_pre(self,rho,states) :

        n = states.shape[0]
        N = (2*rho.Nd+2*rho.Ns)*n
        count = np.zeros(n,dtype=np.int32)
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,states[i])
            if i == 0:
                count[i] = count_0
                L_pre[0:count_0] = L_pre_0
                states_need[0:count_0] = states_need_0
            else:
                count[i] = count[i-1] + count_0
                L_pre[count[i-1]:count[i]] = L_pre_0
                states_need[count[i-1]:count[i]] = states_need_0
        return torch.tensor(count),torch.tensor(L_pre[0:count[-1]],device=self.device),torch.tensor(states_need[0:count[-1]],device=self.device)

    def Lforward_batch_old(self,rho,N = 10000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        the L_rho is calculated in this function 
        the states are calculated in blocks with size N
        _old means for bacnhmark with old method
        '''
        n = self.states.shape[0]
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls_old(rho,self.states[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                # print(count)
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                L_forward_pre = torch.multiply(rho.States(states_need_torch),L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward

    def Lforward_batched_S(self,rho,N = 20000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        the L_rho is calculated in this function 
        the states are calculated in blocks with size N
        don't batch L
        '''
        n = self.states.shape[0]
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        # L_forward_0 = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,self.states[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                # print(count)
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                rho_states = rho.States(states_need_torch)
                L_forward_pre = torch.multiply(rho_states,L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward
            
    def Lforward_nonbatch(self,rho) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        the L_rho is calculated in this function
        '''
        n = self.states.shape[0]
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,self.states[i])
            L_pre_0 = torch.tensor(L_pre_0,device=self.device)
            states_need_0 = torch.tensor(states_need_0,device=self.device)
            L_forward[i] = torch.dot(rho.States(states_need_0),(L_pre_0))
        return L_forward
    
    def Ldforward_batch(self,rho,lmbda=0.,N = 10000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        n = self.states.shape[0]
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.Ldagger_ls(rho,self.states[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                # print(count)
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                L_forward_pre = torch.multiply(rho.States(states_need_torch),L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward

    def LdLforward_nograd_batch(self,rho,lmbda=0.,N = 40000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        n = self.states.shape[0]
        LdL_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.LdaggerL_ls(rho,self.states[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+3*rho.Nd**2 > N or i==n-1 :
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                with torch.no_grad():
                    L_forward_pre = torch.multiply(rho.States(states_need_torch),(L_pre_torch))
                l = 0
                for k in range(j,i+1):
                    LdL_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    # if k==100:
                    #     print(LdL_forward[k])
                    #     print(count_pre[k])
                    #     print(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return LdL_forward.detach()
        
    def LdLforward_nograd(self,rho,lmbda=0.) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        n = self.states.shape[0]
        LdLforward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.LdaggerL_ls_tr(rho,self.states[i])
            L_pre_0 = torch.tensor(L_pre_0,device=self.device)
            states_need_0 = torch.tensor(states_need_0,device=self.device)
            LdLforward[i] = torch.dot(rho.States(states_need_0),(L_pre_0))
            if i==100:
                print(LdLforward[i])
                print(count_0)
        return LdLforward
    
    def Lforward_mc(self,rho,N = 20000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        the L_rho is calculated in this function
        flip and accept according to np.exp(self.lmbda*(Msum1-Msum0))
        '''
        self.flip()
        states_mc = self.states_mc
        # print(states_mc.shape)
        # print(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1))
        weight = torch.tensor(np.exp(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1)),device=self.device)
        # .reshape(-1,1)
        n = self.mc_size
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,states_mc[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                rho_states = rho.States(states_need_torch)
                L_forward_pre = torch.multiply(rho_states,L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward, weight, n

    def Lforward_mc_nonflip(self,rho,N = 20000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        states_mc = self.states_mc
        # print(states_mc.shape)
        # print(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1))
        weight = torch.tensor(np.exp(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1)),device=self.device)
        # .reshape(-1,1)
        n = self.mc_size
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,states_mc[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                rho_states = rho.States(states_need_torch)
                L_forward_pre = torch.multiply(rho_states,L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward, weight, n


    def Lforward_mc_phi(self,rho,N = 20000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        flip and accept according to |rho(s)|^2 
        '''
        states_mc = self.states_mc
        # print(states_mc.shape)
        # print(self.lmbda*np.sum(states_mc[:,0:rho.Nd],axis=1))
        # .reshape(-1,1)
        n = self.mc_size
        L_forward = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        for i in range(0,n):
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,states_mc[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==n-1:
                L_pre_torch = torch.tensor(L_pre[0:count],device=self.device)
                states_need_torch = torch.tensor(states_need[0:count],device=self.device)
                rho_states = rho.States(states_need_torch)
                L_forward_pre = torch.multiply(rho_states,L_pre_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = torch.sum(L_forward_pre[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        return L_forward, n
    
class Liouville_1234(Liouville):
    def __init__(self,read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.) :
        '''
        read : the information of Hs
        nonmccut : trancation of direct summation part
        allcut : trancation
        notsaving = 0(1): (not) saving L
        '''
        super.__init__(read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.)
        self.eta_tr = np.reshape(np.transpose(np.reshape
                        (self.eta,(self.nspin,self.nvar,-1,self.nsgn)),(3,1,0,2)),
                        (self.nsgn,-1))
        #nsgn*nvar*nspin*Nb*M 
        self.gamma_tr = np.reshape(np.transpose(np.reshape
                        (self.gamma,(self.nsgn,self.nspin,self.nvar,-1)),(0,2,1,3)),
                        (-1))
        #(0,2,1,3)

    def L_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        suitable to states(0,1,3,2,4)
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                    +self.gamma_tr.dot(state[0:N])
            count += 1
        elif self.sys==1:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                    +self.gamma_tr.dot(state[0:N])\
                    +(-1.0j)*(-0.25*self.J)*((m_0[0]-m_0[1])*(m_0[2]-m_0[3])-(m_1[0]-m_1[1])*(m_1[2]-m_1[3]))
            count += 1
            if np.sum(m_0[0:2])==1 and np.sum(m_0[2:])==1 :
                A = 0.
                if m_0[0]==0:
                    states_need[count,N] = 1
                    states_need[count,N+1] = 0
                    A = 1.j
                else :
                    states_need[count,N] = 0
                    states_need[count,N+1] = 1
                    A = -1.j
                if m_0[2]==0:
                    states_need[count,N+2] = 1
                    states_need[count,N+3] = 0
                    A *= 1.j
                else :
                    states_need[count,N+2] = 0
                    states_need[count,N+3] = 1
                    A *= -1.j
                L_pre[count] = (-1.0j)*(-0.25*self.J*(1+A))
                count += 1
            if np.sum(m_1[0:2])==1 and np.sum(m_1[2:])==1 :
                A = 0.
                if m_1[0]==0:
                    states_need[count,N] = 1
                    states_need[count,N+1] = 0
                    A = -1.j
                else :
                    states_need[count,N] = 0
                    states_need[count,N+1] = 1
                    A = 1.j
                if m_1[2]==0:
                    states_need[count,N+2] = 1
                    states_need[count,N+3] = 0
                    A *= -1.j
                else :
                    states_need[count,N+2] = 0
                    states_need[count,N+3] = 1
                    A *= 1.j
                L_pre[count] = (1.0j)*(-0.25*self.J*(1+A))
                count += 1
        else :
            print(f"we don't support this type of system: {self.sys}")
            count += 1

        
        i_min,i_min_1 = Nminus,Nminus
        i_pos,i_pos_1 = Npositive,Npositive
        l_0 = 0
        l_1 = 0
        i_0,i_0_1 = 0,0
        i_1,i_1_1 = 0,0   
        #consider nminus/npositive as a Ns*(Nb*M) matrix and calculate L_ls row-wisely
        #i_min/i_pos: the sum of elements in nminus/npos whose indices are larger than i
        #i_0/i_1: the sum of elements in nminus/npos whose indices are smaller than i
        #l_0/l_1: the sum of elements in m_0/m_1 whose indices are smaller than i
        for v in range(0,M):
            #l_0 = np.sum(m_0[0:v])
            l_0 = l_0 + m_0[v]
            if m_0[v] == 0 : 
                for m in range(0,env_num):
                # #5th
                    k = v*env_num+m
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+v] = 1
                        L_pre[count] = (-1.j)*pow(-1,l_0+i_0_1+Nminus)*self.eta_tr[0,k]
                        # 
                        # (-1.j)*pow(-1,i_0_1+l_0+Nminus)*self.eta_tr[0,k]
                        count += 1
                #3th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+v] = 1           
                            L_pre[count] = (-1.0j)*pow(-1,l_0+i_1_1+Nminus+Npositive)
                            # l_0+
                            # (-1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive)
                            count += 1
                i_0_1 = i_0
                i_1_1 = i_1
            if m_0[v]==1 :
                for m in range(0,env_num):
                #1th
                    k = v*env_num+m
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+v] = 0
                            L_pre[count] = (-1.0j)*pow(-1,i_min_1+l_0-1)
                            count += 1
        # #         #7th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+v] = 0
                        L_pre[count] = (-1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k]
                        # (-1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k]
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos
            l_1 = l_1 + m_1[v]
            if m_1[v]==0 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #2th
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+rho.Ns+v] = 1
                            L_pre[count] = -(-1.0j)*pow(-1,i_min_1+l_1+Nminus+Npositive)
                            count += 1
        # #             #8th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] =-(-1.j)*pow(-1,i_pos_1+l_1+Npositive-1)*(self.eta_tr[0,k]).conj()
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos       
            if m_1[v]==1 :
                for m in range(0,env_num):
                    k = v*env_num+m
        # #             #6th
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = -(-1.j)*pow(-1,i_0_1+l_1+Npositive)*(self.eta_tr[1,k]).conj()
                        count += 1
        # # #             #4th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+rho.Ns+v] = 0
                            L_pre[count] = -(-1.0j)*pow(-1,i_1_1+l_1-1)
                            count += 1
            i_0 +=  np.sum(nminus[v*env_num:(v+1)*env_num])
            i_1 +=  np.sum(npositive[v*env_num:(v+1)*env_num])
            i_min -= np.sum(nminus[v*env_num:(v+1)*env_num])
            i_pos -= np.sum(npositive[v*env_num:(v+1)*env_num])
            i_0_1 = i_0
            i_1_1 = i_1
            i_min_1 = i_min
            i_pos_1 = i_pos
        return count, L_pre[0:count], states_need[0:count]

    def L_ls_tr(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+1,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+1,rho.nstate+5),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            if not(Nall>self.allcut):
                L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                        self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
                 +self.gamma_tr.dot(state[0:N])
        else :
            print(f"we don't support this type of system: {self.sys}")
        count += 1

      
        i_min,i_min_1 = Nminus,Nminus
        i_pos,i_pos_1 = Npositive,Npositive
        l_0 = 0
        l_1 = 0
        i_0,i_0_1 = 0,0
        i_1,i_1_1 = 0,0   
        for v in range(0,M):
            #l_0 = np.sum(m_0[0:v])
            l_0 = l_0 + m_0[v]
            if m_0[v] == 0 : 
                for m in range(0,env_num):
                # #5th
                    k = v*env_num+m
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+v] = 1
                        L_pre[count] = (-1.j)*pow(-1,i_0_1+l_0+Nminus)*self.eta_tr[0,k]
                        # (-1.j)*pow(-1,i_0_1+l_0+Nminus)*self.eta_tr[0,k]
                        count += 1
        # #         #3th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+v] = 1           
                            L_pre[count] = (-1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive)
                            # (-1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive)
                            count += 1
                i_0_1 = i_0
                i_1_1 = i_1
            if m_0[v]==1 :
                for m in range(0,env_num):
                #1th
                    k = v*env_num+m
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+v] = 0
                            L_pre[count] = (-1.0j)*pow(-1,i_min_1+l_0-1)
                            count += 1
        # #         #7th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+v] = 0
                        L_pre[count] = (-1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k]
                        # (-1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k]
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos
            l_1 = l_1 + m_1[v]
            if m_1[v]==0 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #2th
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+rho.Ns+v] = 1
                            L_pre[count] = -(-1.0j)*pow(-1,i_min_1+l_1+Nminus+Npositive)
                            count += 1
        # #             #8th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] =-(-1.j)*pow(-1,i_pos_1+l_1+Npositive-1)*(self.eta_tr[0,k]).conj()
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos       
            if m_1[v]==1 :
                for m in range(0,env_num):
                    k = v*env_num+m
        # #             #6th
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = -(-1.j)*pow(-1,i_0_1+l_1+Npositive)*(self.eta_tr[1,k]).conj()
                        count += 1
        # # #             #4th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+rho.Ns+v] = 0
                            L_pre[count] = -(-1.0j)*pow(-1,i_1_1+l_1-1)
                            count += 1
            i_0 +=  np.sum(nminus[v*env_num:(v+1)*env_num])
            i_1 +=  np.sum(npositive[v*env_num:(v+1)*env_num])
            i_min -= np.sum(nminus[v*env_num:(v+1)*env_num])
            i_pos -= np.sum(npositive[v*env_num:(v+1)*env_num])
            i_0_1 = i_0
            i_1_1 = i_1
            i_min_1 = i_min
            i_pos_1 = i_pos
        return count, L_pre[0:count], states_need[0:count]
        
    def H_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|H|s'> and the corresponding states s'
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(rho.Nd+1,dtype=np.complex128)
        states_need = np.zeros((rho.Nd+1,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            L_pre[count] = (-1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                    self.U*np.sum(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
            +self.gamma_tr.dot(state[0:N])
        else :
            print(f"we don't support this type of system: {self.sys}")
        count += 1
        return count, L_pre[0:count], states_need[0:count]
    
    def Ldagger_ls_old(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L\dagger|s'> and the corresponding states s'
        the order of Nd in input state is the same as the version 1.0 method, 
        so the running speed is slower than the L_ls method
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nob = rho.Ns
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            L_pre[count] = (1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                    self.U*(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
            +np.conjugate(self.gamma.dot(state[0:N]))
        else :
            print(f"we don't support this type of system: {self.sys}")
        count += 1

        i = Nminus
        A = nminus
        v = 0        
        for k in range(0,N_half) :
            i = i - nminus[k]
            if nminus[k] == 0 :
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                if m_0[v] == 1 :
                    l = np.sum(m_0[0:v])
                    states_need[count,k] = 1
                    states_need[count,N+v] = 0
                    L_pre[count] = (1.0j)*pow(-1,i+l)
                    count += 1
                if m_1[v] == 0 :
                    l = np.sum(m_1[0:v])
                    states_need[count,k] = 1
                    states_need[count,N+rho.Ns+v] = 1
                    L_pre[count] = (-1.0j)*pow(-1,i+l+Nminus+Npositive)
                    count += 1
        i = 0
        A = npositive
        v = 0
        for k in range(0,N_half) :
            if k > 0 :
                i = i + npositive[k-1]
            if npositive[k] == 0 :
                v = Jtov(k,env_num,rho.No,rho.Nv) 
                if m_0[v] == 0 :
                    states_need[count,N_half+k] = 1
                    states_need[count,N+v] = 1
                    l = np.sum(m_0[0:v])              
                    L_pre[count] =(1.0j)*pow(-1,i+l+Nminus+Npositive)
                    count += 1
                if m_1[v] == 1 :
                    states_need[count,N_half+k] = 1
                    states_need[count,N+rho.Ns+v] = 0
                    l = np.sum(m_1[0:v])
                    L_pre[count] = (-1.0j)*pow(-1,i+l)
                    count += 1
        i = 0
        v = 0
        for k in range(0,N_half) :
            if k > 0 :
                i = i + nminus[k-1]
            if nminus[k] == 1:
                v = Jtov(k,env_num,rho.No,rho.Nv)
                if m_0[v] == 0 :
                    states_need[count,k] = 0
                    states_need[count,N+v] = 1
                    l = np.sum(m_0[0:v])
                    L_pre[count] = (1.j)*pow(-1,i+l+Nminus-1)*(self.eta[k,0]).conj()
                    count += 1
                if m_1[v] == 1 :
                    states_need[count,k] = 0
                    states_need[count,N+rho.Ns+v] = 0
                    l = np.sum(m_1[0:v])
                    L_pre[count] = (-1.j)*pow(-1,i+l+Npositive)*(self.eta[k,1])
                    count += 1
        i = Npositive
        for k in range(0,N_half) :
            i = i - npositive[k]
            if npositive[k] == 1 :
                v = Jtov(k,env_num,rho.No,rho.Nv)
                if m_0[v] == 1 :
                    states_need[count,N_half+k] = 0
                    states_need[count,N+v] = 0
                    l = np.sum(m_0[0:v])
                    L_pre[count] =(1.j)*pow(-1,i+l+Nminus)*(self.eta[k,1]).conj()
                    count += 1
                if m_1[v] == 0 :
                    states_need[count,N_half+k] = 0
                    states_need[count,N+rho.Ns+v] = 1
                    l = np.sum(m_1[0:v])
                    L_pre[count] =(-1.j)*pow(-1,i+l+Npositive-1)*self.eta[k,0]
                    count += 1
        return count, L_pre[0:count], states_need[0:count]

    def Ldagger_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            L_pre[count] = (1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                    self.U*(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
            +np.conjugate(self.gamma_tr.dot(state[0:N]))
        else :
            print(f"we don't support this type of system: {self.sys}")
        count += 1

      
        i_min,i_min_1 = Nminus,Nminus
        i_pos,i_pos_1 = Npositive,Npositive
        l_0 = 0
        l_1 = 0
        i_0,i_0_1 = 0,0
        i_1,i_1_1 = 0,0   
        for v in range(0,M):
            #l_0 = np.sum(m_0[0:v])
            l_0 = l_0 + m_0[v]
            if m_0[v] == 0 : 
                for m in range(0,env_num):
                #5th
                    k = v*env_num+m
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+v] = 1
                        L_pre[count] = (1.0j)*pow(-1,i_min_1+l_0)
                        count += 1
                #3th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+v] = 1           
                            L_pre[count] = (1.j)*pow(-1,i_pos_1+l_0+Nminus)*self.eta_tr[1,k].conj()
                            # (1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k].conj()
                            count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos
            if m_0[v]==1 :
                for m in range(0,env_num):
                #1th
                    k = v*env_num+m
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+v] = 0
                            L_pre[count] = (1.j)*pow(-1,i_0_1+l_0+Nminus-1)*(self.eta_tr[0,k]).conj()
                            # (1.j)*pow(-1,i_0_1+l_0+Nminus)*(self.eta_tr[0,k]).conj()
                            count += 1
        #         #7th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+v] = 0
                        L_pre[count] =(1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive-1)
                        # (1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive-1)
                        count += 1
                i_0_1 = i_0
                i_1_1 = i_1
            l_1 = l_1 + m_1[v]
            if m_1[v]==0 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #2th
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 0:
                        if ifplus:
                            states_need[count,k] = 1
                            states_need[count,N+rho.Ns+v] = 1
                            L_pre[count] = (-1.j)*pow(-1,i_0_1+l_1+Npositive)*(self.eta_tr[1,k])
                            count += 1
        #             #8th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] = (-1.0j)*pow(-1,i_1_1+l_1-1)
                        count += 1
                i_0_1 = i_0
                i_1_1 = i_1      
            if m_1[v]==1 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #6th
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = (-1.0j)*pow(-1,i_min_1+l_1+Nminus+Npositive)
                        count += 1
        #             #4th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 0:
                        if ifplus:
                            states_need[count,N_half+k] = 1
                            states_need[count,N+rho.Ns+v] = 0
                            L_pre[count] = (-1.j)*pow(-1,i_pos_1+l_1+Npositive-1)*(self.eta_tr[0,k])
                            count += 1
            i_0 +=  np.sum(nminus[v*env_num:(v+1)*env_num])
            i_1 +=  np.sum(npositive[v*env_num:(v+1)*env_num])
            i_min -= np.sum(nminus[v*env_num:(v+1)*env_num])
            i_pos -= np.sum(npositive[v*env_num:(v+1)*env_num])
            i_0_1 = i_0
            i_1_1 = i_1
            i_min_1 = i_min
            i_pos_1 = i_pos
        return count, L_pre[0:count], states_need[0:count]
    
    def Ldagger_ls_nocut(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L|s'> and the corresponding states s'
        '''
        global num_energy
        N = rho.Nd
        N_half = N//2
        M = rho.Ns
        num_energy = M
        env_num = rho.Nb*rho.M
        nminus = state[0:N_half]
        Nminus = np.sum(nminus)
        npositive = state[N_half:N]
        Npositive = np.sum(npositive)
        Nall = Npositive+Nminus
        ifplus = Nall < self.allcut
        m_0 = state[N:N+M]
        m_1 = state[N+M:N+M*2]
        count = 0
        L_pre = np.zeros(2*rho.Nd+2*rho.Ns,dtype=np.complex128)
        states_need = np.zeros((2*rho.Nd+2*rho.Ns,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        # states_need[count] = state
        if self.sys==0:
            L_pre[count] = (1.0j)*(np.sum(self.energy_level*(m_0-m_1))+ 
                                    self.U*(np.prod(m_0.reshape(-1,2),axis=1)-np.prod(m_1.reshape(-1,2),axis=1))) \
            +np.conjugate(self.gamma_tr.dot(state[0:N]))
        else :
            print(f"we don't support this type of system: {self.sys}")
        count += 1

      
        i_min,i_min_1 = Nminus,Nminus
        i_pos,i_pos_1 = Npositive,Npositive
        l_0 = 0
        l_1 = 0
        i_0,i_0_1 = 0,0
        i_1,i_1_1 = 0,0   
        for v in range(0,M):
            #l_0 = np.sum(m_0[0:v])
            l_0 = l_0 + m_0[v]
            if m_0[v] == 0 : 
                for m in range(0,env_num):
                #5th
                    k = v*env_num+m
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+v] = 1
                        L_pre[count] = (1.0j)*pow(-1,i_min_1+l_0)
                        count += 1
                #3th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 0:
                        states_need[count,N_half+k] = 1
                        states_need[count,N+v] = 1           
                        L_pre[count] = (1.j)*pow(-1,i_pos_1+l_0+Nminus)*self.eta_tr[1,k].conj()
                        # (1.j)*pow(-1,i_pos_1+l_0+Nminus-1)*self.eta_tr[1,k].conj()
                        count += 1
                i_min_1 = i_min
                i_pos_1 = i_pos
            if m_0[v]==1 :
                for m in range(0,env_num):
                #1th
                    k = v*env_num+m
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 0:
                        states_need[count,k] = 1
                        states_need[count,N+v] = 0
                        L_pre[count] = (1.j)*pow(-1,i_0_1+l_0+Nminus-1)*(self.eta_tr[0,k]).conj()
                        # (1.j)*pow(-1,i_0_1+l_0+Nminus)*(self.eta_tr[0,k]).conj()
                        count += 1
        #         #7th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+v] = 0
                        L_pre[count] =(1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive-1)
                        # (1.0j)*pow(-1,i_1_1+l_0+Nminus+Npositive-1)
                        count += 1
                i_0_1 = i_0
                i_1_1 = i_1
            l_1 = l_1 + m_1[v]
            if m_1[v]==0 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #2th
                    i_0_1 = i_0_1 + nminus[k]
                    if nminus[k] == 0:
                        states_need[count,k] = 1
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] = (-1.j)*pow(-1,i_0_1+l_1+Npositive)*(self.eta_tr[1,k])
                        count += 1
        #             #8th
                    i_1_1 = i_1_1 + npositive[k]
                    if npositive[k] == 1:
                        states_need[count,N_half+k] = 0
                        states_need[count,N+rho.Ns+v] = 1
                        L_pre[count] = (-1.0j)*pow(-1,i_1_1+l_1-1)
                        count += 1
                i_0_1 = i_0
                i_1_1 = i_1      
            if m_1[v]==1 :
                for m in range(0,env_num):
                    k = v*env_num+m
                    #6th
                    i_min_1 = i_min_1 - nminus[k]
                    if nminus[k] == 1:
                        states_need[count,k] = 0
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = (-1.0j)*pow(-1,i_min_1+l_1+Nminus+Npositive)
                        count += 1
        #             #4th
                    i_pos_1 = i_pos_1 - npositive[k]
                    if npositive[k] == 0:
                        states_need[count,N_half+k] = 1
                        states_need[count,N+rho.Ns+v] = 0
                        L_pre[count] = (-1.j)*pow(-1,i_pos_1+l_1+Npositive-1)*(self.eta_tr[0,k])
                        count += 1
            i_0 +=  np.sum(nminus[v*env_num:(v+1)*env_num])
            i_1 +=  np.sum(npositive[v*env_num:(v+1)*env_num])
            i_min -= np.sum(nminus[v*env_num:(v+1)*env_num])
            i_pos -= np.sum(npositive[v*env_num:(v+1)*env_num])
            i_0_1 = i_0
            i_1_1 = i_1
            i_min_1 = i_min
            i_pos_1 = i_pos
        return count, L_pre[0:count], states_need[0:count]
    
    def LdaggerL_ls_old(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L\daggerL|s'> and the corresponding states s'
        '''
        count = 0
        L_pre = np.zeros((2*rho.Nd+1)**2,dtype=np.complex128)
        states_need = np.zeros(((2*rho.Nd+1)**2,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        count_0,L_pre_0,states_need_0 = self.Ldagger_ls_old(rho,state)
        for j in range(0,count_0):
            count_1,L_pre_1,states_need_1 = self.L_ls_old(rho,states_need_0[j])
            L_pre[count:count+count_1] = L_pre_0[j]*L_pre_1
            states_need[count:count+count_1,:] = states_need_1
            count += count_1
        return count, L_pre[0:count], states_need[0:count]
    
    def LdaggerL_ls(self,rho,state) :
        '''
        input: state s
        output: the all non-zero <s|L\daggerL|s'> and the corresponding states s'
        '''
        count = 0
        L_pre = np.zeros((2*rho.Nd+1)**2,dtype=np.complex128)
        states_need = np.zeros(((2*rho.Nd+1)**2,rho.nstate),dtype=np.float64)
        
        states_need[:,:] = state
        count_0,L_pre_0,states_need_0 = self.Ldagger_ls(rho,state)
        for j in range(0,count_0):
            count_1,L_pre_1,states_need_1 = self.L_ls(rho,states_need_0[j])
            L_pre[count:count+count_1] = L_pre_0[j]*L_pre_1
            states_need[count:count+count_1,:] = states_need_1
            count += count_1
        return count, L_pre[0:count], states_need[0:count]


    def set_states(self,states):
        '''
        translate old states (read from table file) to new states (required to perfrom L_ls)
        states are read from table.data
        '''
        n = states.shape[0]
        nstate = states.shape[1]
        self.states_pre = np.zeros((n,nstate),dtype=np.float64)
        self.states_pre[:,:] = states
        # .to('cpu').numpy()
        self.states = self.states_pre

        states_pre = np.reshape(np.transpose(np.reshape
                (states[:,0:self.Nd],(n,self.nsgn,self.nspin,self.nvar,-1)),(0,1,3,2,4)),
                (n,-1))
        # # 0,1,3,2,4
        self.states[:,:self.Nd] = states_pre

        # self.states = np.reshape(np.transpose(np.reshape
        #                 (self.states,(self.nsgn,self.nspin,self.nvar,-1,nstate+5)),(0,2,1,3,4)),
        #                 (-1,nstate+5))
        if self.nonmccut<self.allcut:
            i = np.random.randint(0,n,(self.mc_size))
            self.states_mc = self.states[i].copy()
            self.batch_init()
            self.cal_mccoe()
        return 0

    def batch_init(self):
        '''
        initialize the states of MC sampling to satisfy the rank of all states is larger than mccut
        '''
        m =  self.Nd
        count = 0
        k = 0
        # flags = np.zeros_like(self.states_mc)
        #lambda不能设置的很负
        while(not(np.all(np.sum(self.states_mc[:,0:self.Nd],axis=1)>self.nonmccut))):
            # print(f'k {k}')
            # k = k+1
            flags = np.zeros_like(self.states_mc)
            flagsm = np.zeros_like(self.states_mc)
            flagsn = np.zeros_like(self.states_mc)
            u = np.random.rand((1))
            if u[0]<0.5:
                #flip n and the relative m, which are different from each other
                #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
                m_flip = np.random.randint(0,self.Nd,(self.mc_size))
                n_flip = m_flip//self.env
                a = np.eye(self.nstate)
                flagsm[:,:self.nstate] =  a[m_flip]
                flagsn[:,:self.nstate] =  a[m+n_flip]
                flags = flagsm + flagsn 
                flag_diff = np.abs(np.sum((flagsm-flagsn)*self.states_mc,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.states_mc)
                s1 = self.states_mc + ds
            else :
                #flip n and the relative m, which are equal to each other
                #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
                m_flip = np.random.randint(0,self.Nd,(self.mc_size))
                n_flip = m_flip//self.env
                n_flip = (1 - n_flip//self.Ns)*self.Ns+n_flip%self.Ns
                a = np.eye(self.nstate)
                flagsm[:,:self.nstate] =  a[m_flip]
                flagsn[:,:self.nstate] =  a[m+n_flip]
                flags = flagsm + flagsn 
                flag_diff = 1-np.abs(np.sum((flagsm-flagsn)*self.states_mc,axis=1)).reshape(-1,1)
                ds = flag_diff*flags*(1-2*self.states_mc)
                s1 =self.states_mc + ds
            Msum1 = np.sum(s1[:,0:self.Nd],axis=1).reshape(-1,1)
            Msum0 = np.sum(self.states_mc[:,0:self.Nd],axis=1).reshape(-1,1)
            accept_allcut = np.less_equal(Msum1,self.allcut)
            accept_greater = np.greater(Msum1,Msum0)
            self.states_mc = self.states_mc + \
                accept_greater*accept_allcut*ds
        return self.states_mc

    def flip(self):
        '''
         flip and accept according to np.exp(self.lmbda*(Msum1-Msum0))
        '''
        m =  self.Nd
        count = 0
        k = 0
        flags = np.zeros_like(self.states_mc)
        flagsm = np.zeros_like(self.states_mc)
        flagsn = np.zeros_like(self.states_mc)
        #lambda不能设置的很负
        u = np.random.rand((2))
        if u[0] < 1/self.env :
            #flip两个m
            n_flip = np.random.randint(0,self.Ns,(self.mc_size))
            a = np.eye(self.nstate)
            flagsm[:,:self.nstate] =  a[m+n_flip]
            flagsn[:,:self.nstate] =  a[m+self.Ns+n_flip]
            flags = flagsm + flagsn 
            flag_diff = 1-np.abs(np.sum((flagsm-flagsn)*self.states_mc,axis=1)).reshape(-1,1)
            ds = flag_diff*flags*(1-2*self.states_mc)
            s1 = self.states_mc + ds
        elif u[1]<0.5:
        #flip n and the relative m, which are different from each other
        #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
            #flip n and the relative m, which are different from each other
            #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
            m_flip = np.random.randint(0,self.Nd,(self.mc_size))
            n_flip = m_flip//self.env
            a = np.eye(self.nstate)
            flagsm[:,:self.nstate] =  a[m_flip]
            flagsn[:,:self.nstate] =  a[m+n_flip]
            flags = flagsm + flagsn 
            flag_diff = np.abs(np.sum((flagsm-flagsn)*self.states_mc,axis=1)).reshape(-1,1)
            ds = flag_diff*flags*(1-2*self.states_mc)
            s1 = self.states_mc + ds
        else :
            #flip n and the relative m, which are equal to each other
            #nsgn*nvar*nspin*Nb*M to nsgn*nvar*nspin
            m_flip = np.random.randint(0,self.Nd,(self.mc_size))
            n_flip = m_flip//self.env
            n_flip = (1 - n_flip//self.Ns)*self.Ns+n_flip%self.Ns
            a = np.eye(self.nstate)
            flagsm[:,:self.nstate] =  a[m_flip]
            flagsn[:,:self.nstate] =  a[m+n_flip]
            flags = flagsm + flagsn 
            flag_diff = 1-np.abs(np.sum((flagsm-flagsn)*self.states_mc,axis=1)).reshape(-1,1)
            ds = flag_diff*flags*(1-2*self.states_mc)
            s1 =self.states_mc + ds
        Msum1 = np.sum(s1[:,0:self.Nd],axis=1).reshape(-1,1)
        # print(Msum1.shape)
        Msum0 = np.sum(self.states_mc[:,0:self.Nd],axis=1).reshape(-1,1)
        accept_mc = np.less(np.random.rand(self.mc_size,1),np.exp(self.lmbda*(Msum1-Msum0)))
        accept_allcut = np.less_equal(Msum1,self.allcut)
        accept_nonmccut = np.greater(Msum1,self.nonmccut)
        # print(accept_allcut.shape)
        # print(accept_nonmccut.shape)
        # print(accept_mc.shape)
        accept = accept_mc*accept_nonmccut*accept_allcut
        # print(f'accept: {np.sum(accept)}')
        # print(f'accept_allcut: {np.sum(accept_allcut)}')
        # print(f'accept_nonmccut: {np.sum(accept_nonmccut)}')
        self.states_mc = self.states_mc + \
            accept*ds
        # print(f'accept {np.sum(accept)}')
        return self.states_mc

    
def AcceptProbability(Statenow,Statenext,rho,lamda) :
#the probability of acceptance
    if rho.f0(Statenext):
        N_now = np.sum(Statenow[0:rho.Nd])
        N_next = np.sum(Statenext[0:rho.Nd])
        a = torch.exp(lamda*(N_next-N_now))
        return torch.min(torch.tensor([1.,a]))

    return 0.




def SubscriptTrans0(m) :
    #H(m,n) and H(dot(m),dot(k));for
    #two to ten
    j = np.zeros(1,dtype=int)
    a = np.zeros(1,dtype=int)
    N = m.size
    for i in m :
        a = a + i*torch.pow(2,N-j-1)
        j = j + 1
    return a

def SubscriptTrans0_torch(m) :
    shape = m.size()[-1]
    a = torch.zeros(shape,dtype=torch.int32,device=m.device)
    for i in range(shape) : 
        a[i] = torch.pow(2,torch.tensor([shape-i-1]))
    return torch.sum(m*a,dim=-1)

def SubscriptTrans1(a,N) :
    #ten to two
    m = np.zeros(N,dtype=int)
    i = 0
    while a > 0 :
        m[N-i-1] = np.mod(a,2)
        a = int(np.div(a,2,rounding_mode='floor'))
        i += 1
    return m




def LeftRho(rho,nminus,npositive,m_1) :
    global num_energy
    N = torch.pow(2,num_energy)
    Rho = np.zeros(N,dtype=np.complex128)
    for i in range(0,N) :
        Rho[i] = rho.State(torch.hstack((nminus,npositive,SubscriptTrans1(i,num_energy),m_1)))
    return Rho

def RightRho(rho,nminus,npositive,m_0) :
    global num_energy
    Rho = np.zeros(2**num_energy,dtype=np.complex128)
    for i in range(0,2**num_energy) :
        Rho[i] = rho.State(torch.hstack((nminus,npositive,m_0,SubscriptTrans1(i,num_energy))))
    return Rho

def Jtov(k,env_num,No,Nv) :
    # Nd: nsgn,No,Nv,Nb,M  ;c.data: Nv,No
    # No is the index of spin
    k = k%(env_num*No*Nv)//env_num
    return No*(k%Nv)+k//Nv


    

def is_same_0(state,state1) :
    if all(state == state1) :
        return True
    else :
        return False

def is_same_minusjadd(state,state1,i) :
    N = state.size
    k = 0
    if any(state[N//2:N] != state1[N//2:N]) :
        return False
    for j in range(0,N//2) :
        if state[j] != state1[j] :
            k = k + 1
            i[0] = j
        if k == 2 :
            return False
    if k == 1 :
        if state[i[0]] == 0 :
            return True
        else :
            return False
    else :
        return False

def is_same_addjadd(state,state1,i) :
    N = state.size
    k = 0
    if any(state[0:N//2] != state1[0:N//2]) :
        return False
    for j in range(0,N//2) :
        if state[N//2+j] != state1[N//2+j] :
            k = k + 1
            i[0] = j
        if k == 2 :
            return False
    if k == 1 :
        if state[N//2+i[0]] == 0 :
            return True
        else :
            return False
    else :
        return False

def is_same_minusjminus(state,state1,i) :
    N = state.size
    k = 0
    if any(state[N//2:N] != state1[N//2:N]) :
        return False
    for j in range(0,N//2) :
        if state[j] != state1[j] :
            k = k + 1
            i[0] = j
        if k == 2 :
            return False
    if k == 1 :
        if state[i[0]] == 1 :
            return True
        else :
            return False
    else :
        return False

def is_same_addjminus(state,state1,i) :
    N = state.size
    k = 0
    if any(state[0:N//2] != state1[0:N//2]) :
        return False
    for j in range(0,N//2) :
        if state[N//2+j] != state1[N//2+j] :
            k = k + 1
            i[0] = j
        if k == 2 :
            return False
    if k == 1 :
        if state[N//2+i[0]] == 1 :
            return True
        else :
            return False
    else :
        return False
