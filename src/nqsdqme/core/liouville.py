import numpy as np
import scipy.sparse as sp
import time
import sys as sys_sys
import torch
from scipy.special import comb
from ..global_defs import get_device

def Jtov(k,env_num,No,Nv) :
    # Nd: nsgn,No,Nv,Nb,M  ;c.data: Nv,No
    # No is the index of spin
    k = k%(env_num*No*Nv)//env_num
    return No*(k%Nv)+k//Nv

class Liouville():
    def __init__(self,read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.) :
        '''
        read : the information of Hs
        nonmccut : trancation of direct summation part
        allcut : trancation
        notsaving = 0(1): (not) saving L
        '''
        self.device = device
        self.sys_mode = read[0]
        if self.sys_mode==0:
            self.energy_level = read[1]
            self.U = read[2]
        elif self.sys_mode==1: 
            self.energy_level = read[1]
            self.U = read[2]
            self.J = read[10]
        else:
            print(f"we don't support this type of system: {self.sys_mode}")
        self.gamma = read[3]
        self.eta = read[4]
        self.nsgn = read[9]
        self.nspin = read[6]
        self.nvar = read[5] #the number of system energy levels (oen level has two spins)
        self.nalf = read[8] #the number of baths
        self.ncor = read[7] #the number of spectral decomposition points
        self.Nd = self.nsgn*self.nvar*self.nspin*self.ncor*self.nalf
        self.Nbs = self.nvar*self.nspin*self.ncor*self.nalf
        self.Ns = self.nvar*self.nspin
        self.env = self.ncor*self.nalf
        self.nstate = self.Nd + 2*self.Ns
        self.nrho = pow(2,self.nspin*self.nvar)
        self.nonmccut = nonmccut
        self.allcut = allcut
        self.notsaving = notsaving
        self.mc_size = mc_size
        self.lmbda = lmbda

        #lmbda = 0 can ensure the most fliping
        # add chemical potential
        hbar = 0.658211928
        # hbar = 1.0
        chem_potential = True
        if chem_potential :
            chem_potentials = np.loadtxt("chem_potentials.data")
            chem_potentials = chem_potentials.reshape((self.nalf,self.nspin))
            mu = np.zeros((self.nsgn,self.nspin,self.nvar,self.nalf,self.ncor), dtype=np.complex128)
            for a1 in range(self.nsgn) :
                for a2 in range(self.nspin) :
                    for a3 in range(self.nvar) :
                        for a4 in range(self.nalf) :
                            for a5 in range(self.ncor) :
                                mu[a1,a2,a3,a4,a5] = 1.j*(-1)**(a1+1)*chem_potentials[a4,a2]
            self.gamma = self.gamma + mu.flatten()/hbar
        #nsgn,nvar*nspin*Nb*M    
        #the indices of the input of rho corresponds to the the indices of self.gamma,
        #which is nsgn*nspin*nvar*Nb*M 
        # self.eta_tr = np.reshape(np.transpose(np.reshape
        #                 (self.eta,(self.nspin,self.nvar,-1,self.nsgn)),(3,1,0,2)),
        #                 (self.nsgn,-1))
        self.eta_tr = np.reshape(np.transpose(np.reshape
                        (self.eta,(self.nspin,self.nvar,-1,self.nsgn)),(3,0,1,2)),
                        (self.nsgn,-1))
        #nsgn*nvar*nspin*Nb*M 
        self.gamma_tr = np.reshape(np.transpose(np.reshape
                        (self.gamma,(self.nsgn,self.nspin,self.nvar,-1)),(0,1,2,3)),
                        (-1))
        #(0,2,1,3)
            # Nd: nsgn,No,Nv,Nb,M  ;c.data: Nv,No
    # No is the index of spin

    def Llocals(self,rho,states) :
        return  self.L_ls_batched(rho,states)
    
    def H_ls_bathced(self, rho, states):
        '''
        Fully vectorized version for batched states
        '''
        N = rho.Nd
        N_half = N // 2
        M = rho.Ns
        batch_size = states.shape[0]
        state_length = states.shape[1]
        
        # Extract batched components
        nminus = states[:, 0:N_half]
        npositive = states[:, N_half:N]
        m_0 = states[:, N:N+M]
        m_1 = states[:, N+M:N+M*2]
        
        # Compute sums
        Nminus = np.sum(nminus, axis=1)
        Npositive = np.sum(npositive, axis=1)
        Nall = Npositive + Nminus
        
        # Initialize results
        all_results = []
        
        if self.sys_mode in [0, 1]:
            # Base case for all valid batches
            valid_batches = Nall <= self.allcut
            
            if np.any(valid_batches):
                condition = valid_batches
                batch_indices = np.where(condition)[0]

                n = len(batch_indices)
                # print(n)
                states_need = states[batch_indices].copy()
                
                m_0_batch = m_0[batch_indices]
                m_1_batch = m_1[batch_indices]
                energy_term = np.sum(self.energy_level * (m_0_batch - m_1_batch), axis=1)
                
                m_0_reshaped = m_0_batch.reshape(n, -1, 2)
                m_1_reshaped = m_1_batch.reshape(n, -1, 2)

                U_term = self.U * np.sum(np.prod(m_0_reshaped, axis=2) - 
                                np.prod(m_1_reshaped, axis=2), axis=1)
        
                gamma_term = self.gamma_tr.dot(states_need[:, 0:N].T).T
                
                L_pre =  (-1.0j) * (energy_term + U_term) + gamma_term

                
                # Vectorized computation of base terms
                
                if self.sys_mode == 1:
                    J_term = (-0.25 * self.J) * ((m_0_batch[:, 0] - m_0_batch[:, 1]) * (m_0_batch[:, 2] - m_0_batch[:, 3]) - 
                                            (m_1_batch[:, 0] - m_1_batch[:, 1]) * (m_1_batch[:, 2] - m_1_batch[:, 3]))
                    L_pre += (-1.0j) * J_term

                
                # Add base results
                all_results.append((batch_indices, L_pre, states_need))             
            # Mode 1 specific conditions
            if self.sys_mode == 1:
                # Precompute conditions

                m_0_01_sum = np.sum(m_0[:, 0:2], axis=1)
                m_0_23_sum = np.sum(m_0[:, 2:], axis=1)
                m_1_01_sum = np.sum(m_1[:, 0:2], axis=1)
                m_1_23_sum = np.sum(m_1[:, 2:], axis=1)
                
                # Condition m_0
                cond1 = valid_batches & (m_0_01_sum == 1) & (m_0_23_sum == 1)
      
                condition = cond1 & (m_0[:, 0] != m_0[:, 2]) #A=1
                batch_indices = np.where(condition)[0]

                states_need = states[batch_indices].copy()
                states_need[:,N:N+M] = 1 - states_need[:,N:N+M]

                A = 1
                L_pre = (-1.0j)*(-0.25*self.J*(1+A))*np.ones_like(batch_indices,dtype=np.complex128)

                all_results.append((batch_indices, L_pre, states_need)) 

                # Condition m_1
                cond1 = valid_batches & (m_1_01_sum == 1) & (m_1_23_sum == 1)

                condition = cond1 & (m_1[:, 0] != m_1[:, 2]) #A=1
                batch_indices = np.where(condition)[0]

                states_need = states[batch_indices].copy()
                states_need[:,N+M:N+2*M] = 1 - states_need[:,N+M:N+2*M]

                A = 1
                L_pre = (1.0j)*(-0.25*self.J*(1+A))*np.ones_like(batch_indices,dtype=np.complex128)

                all_results.append((batch_indices, L_pre, states_need))         
        else:
            all_results.append((batch_indices, 0, states[batch_indices].copy()))
            print(f"we don't support this type of system: {self.sys_mode}")
                
        return all_results


    def L_ls_batched(self, rho, states:np.ndarray):
        '''
        Vectorized version for batched states
        input: states array of shape (batch_size, state_length)
        output: batch_counts, batch_L_pre, batch_states_need
        '''
        N = rho.Nd
        N_half = N // 2
        M = rho.Ns
        No = rho.No
        Nv = rho.Nv
        env_num = rho.Nb * rho.M
        batch_size = states.shape[0]
        state_length = states.shape[1]
        
        # Extract batched components
        nminus = states[:, 0:N_half]  # shape: (batch_size, N_half)
        npositive = states[:, N_half:N]  # shape: (batch_size, N_half)
        m_0 = states[:, N:N+M]  # shape: (batch_size, M)
        m_1 = states[:, N+M:N+M*2]  # shape: (batch_size, M)
        
        # Compute sums
        Nminus = np.sum(nminus, axis=1)  # shape: (batch_size,)
        Npositive = np.sum(npositive, axis=1)  # shape: (batch_size,)
        Nall = Npositive + Nminus  # shape: (batch_size,)
        ifplus = Nall < self.allcut  # shape: (batch_size,)
        
        # Precompute cumulative sums for faster indexing
        nminus_cumsum = np.concatenate([np.zeros((batch_size, 1)), 
                                    np.cumsum(nminus[:, :-1], axis=1)], axis=1)
        npositive_cumsum = np.concatenate([np.zeros((batch_size, 1)), 
                                        np.cumsum(npositive[:, :-1], axis=1)], axis=1)
        
        # Precompute reverse cumulative sums (sum of elements from index to end)
        nminus_rev_cumsum = np.concatenate([np.cumsum(nminus[:, ::-1], axis=1)[:, ::-1], 
                                        np.zeros((batch_size, 1))], axis=1)
        npositive_rev_cumsum = np.concatenate([np.cumsum(npositive[:, ::-1], axis=1)[:, ::-1], 
                                            np.zeros((batch_size, 1))], axis=1)
        
        # Precompute l_0 and l_1 for all v values and batches
        c_0_vals = np.zeros((batch_size, M))
        c_1_vals = np.zeros((batch_size, M))


        if self.sys_mode == 0:
            c_0_vals[:, 0] = m_0[:, 1]
            c_1_vals[:, 0] = m_1[:, 1]
        else:
            c_0_vals[:, 0] = np.sum(m_0[:, 2:], axis=1)
            c_1_vals[:, 0] = np.sum(m_1[:, 2:], axis=1)
            c_0_vals[:, 1] = np.sum(m_0[:, 2:], axis=1) + m_0[:, 0]
            c_1_vals[:, 1] = np.sum(m_1[:, 2:], axis=1) + m_1[:, 0]
            c_0_vals[:, 2] = m_0[:, 3]
            c_1_vals[:, 2] = m_1[:, 3]

        
        # Precompute v_tr for all v values
        w_vals = M - np.arange(M) - 1  # shape: (M,)
        v_tr_vals = Nv * (w_vals % No) + w_vals // No  # shape: (M,)
        
        # Initialize results storage
        all_results = []

        
        # Process H_ls for all batches first

        results_H = self.H_ls_bathced(rho, states)
        all_results.extend(results_H)
        
        # Process all v and m combinations
        for v in range(M):
            v_tr = v_tr_vals[v]
            k_start = v_tr * env_num
            k_end = k_start + env_num
            k_indices = np.arange(k_start, k_end)
            
            # Get l_0 and l_1 values for this v across all batches
            c_0_batch = c_0_vals[:, v]
            c_1_batch = c_1_vals[:, v]
            
            # Process m_0[v] conditions
            m_0_v_0 = m_0[:, v] == 0
            m_0_v_1 = m_0[:, v] == 1
            
            # Process m_1[v] conditions
            m_1_v_0 = m_1[:, v] == 0
            m_1_v_1 = m_1[:, v] == 1
            
            # Case 1: m_0[v] == 0, 5th condition
            if np.any(m_0_v_0):
                for k in k_indices:
                    # Find batches where condition is true
                    condition = m_0_v_0 & (nminus[:, k] == 1)
                    batch_indices = np.where(condition)[0]

                    #mutate states to the required
                    states_need = states[batch_indices].copy()
                    states_need[:,k] = 0
                    states_need[:,N + v] = 1

                    #compute the sign and L_val
                    c_0 = c_0_batch[batch_indices] # from the annihilation operator
                    i_0_1 = nminus_cumsum[batch_indices, k] # from the dissipaton 
                    Nminus_val = Nminus[batch_indices]
                    sign = (-1) ** (c_0 + i_0_1 + Nminus_val - 1)

                    L_pre_val = (-1.j) * sign * self.eta_tr[0, k]
                    all_results.append((batch_indices, L_pre_val, states_need))  
            
            # Case 2: m_0[v] == 0, 3rd condition
            if np.any(m_0_v_0 & ifplus):
                for k in k_indices:
                    condition = m_0_v_0 & ifplus & (npositive[:, k] == 0)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,N_half + k] = 1
                    states_need[:,N + v] = 1

                    c_0 = c_0_batch[batch_indices]
                    i_1_1 = npositive_cumsum[batch_indices, k]
                    Nminus_val = Nminus[batch_indices]
                    Npositive_val = Npositive[batch_indices]
                    sign = (-1) ** (c_0 + i_1_1 + Nminus_val + Npositive_val)

                    L_pre_val = (-1.0j) * sign
                    all_results.append((batch_indices, L_pre_val, states_need))

            # Case 3: m_0[v] == 1, 1st condition
            if np.any(m_0_v_1 & ifplus):
                for k in k_indices:
                    condition = m_0_v_1 & ifplus & (nminus[:, k] == 0)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,k] = 1
                    states_need[:,N + v] = 0

                    i_min_1 = nminus_rev_cumsum[batch_indices, k + 1]
                    c_0 = c_0_batch[batch_indices]
                    sign = (-1) ** (i_min_1 + c_0)

                    L_pre_val = (-1.0j) * sign
                    all_results.append((batch_indices, L_pre_val, states_need))  
            
            # Case 4: m_0[v] == 1, 7th condition
            if np.any(m_0_v_1):
                for k in k_indices:
                    condition = m_0_v_1 & (npositive[:, k] == 1)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,N_half + k] = 0
                    states_need[:,N + v] = 0

                    c_0 = c_0_batch[batch_indices]
                    i_pos_1 = npositive_rev_cumsum[batch_indices, k + 1]
                    Nminus_val = Nminus[batch_indices]
                    
                    sign = (-1) ** (i_pos_1 + c_0 + Nminus_val)

                    L_pre_val = (-1.j) * sign * self.eta_tr[1, k]
                    all_results.append((batch_indices, L_pre_val, states_need))
            
            # Case 5: m_1[v] == 0, 2nd condition
            if np.any(m_1_v_0 & ifplus):
                for k in k_indices:
                    condition = m_1_v_0 & ifplus & (nminus[:, k] == 0)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,k] = 1
                    states_need[:,N + M + v] = 1

                    c_1 = c_1_batch[batch_indices]
                    i_min_1 = nminus_rev_cumsum[batch_indices, k + 1]
                    Nminus_val = Nminus[batch_indices]
                    Npositive_val = Npositive[batch_indices]
                    
                    sign = (-1) ** (i_min_1 + c_1 + Nminus_val + Npositive_val)
                    L_pre_val = -(-1.0j) * sign
                    all_results.append((batch_indices, L_pre_val, states_need))
            
            # Case 6: m_1[v] == 0, 8th condition
            if np.any(m_1_v_0):
                for k in k_indices:
                    condition = m_1_v_0 & (npositive[:, k] == 1)
                    batch_indices = np.where(condition)[0]
                    

                    states_need = states[batch_indices].copy()
                    states_need[:,N_half + k] = 0
                    states_need[:,N + M + v] = 1

                    c_1 = c_1_batch[batch_indices]
                    i_pos_1 = npositive_rev_cumsum[batch_indices, k + 1] 
                    Npositive_val = Npositive[batch_indices]
                    
                    sign = (-1) ** (i_pos_1 + c_1 + Npositive_val - 1)
                    L_pre_val = -(-1.j) * sign * self.eta_tr[0, k].conj()
                    all_results.append((batch_indices, L_pre_val, states_need))
            
            # Case 7: m_1[v] == 1, 6th condition
            if np.any(m_1_v_1):
                for k in k_indices:
                    condition = m_1_v_1 & (nminus[:, k] == 1)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,k] = 0
                    states_need[:,N + M + v] = 0

                    c_1 = c_1_batch[batch_indices]
                    i_0_1 = nminus_cumsum[batch_indices, k]
                    Npositive_val = Npositive[batch_indices]
                    
                    sign = (-1) ** (i_0_1 + c_1 + Npositive_val)
                    L_pre_val = -(-1.j) * sign * self.eta_tr[1, k].conj()
                    all_results.append((batch_indices, L_pre_val, states_need))
            
            # Case 8: m_1[v] == 1, 4th condition
            if np.any(m_1_v_1 & ifplus):
                for k in k_indices:
                    condition = m_1_v_1 & ifplus & (npositive[:, k] == 0)
                    batch_indices = np.where(condition)[0]
                    
                    states_need = states[batch_indices].copy()
                    states_need[:,N_half + k] = 1
                    states_need[:,N + M + v] = 0

                    c_1 = c_1_batch[batch_indices]
                    i_1_1 = npositive_cumsum[batch_indices, k]
                    sign = (-1) ** (i_1_1 + c_1)

                    L_pre_val = -(-1.0j) * sign
                        
                    all_results.append((batch_indices, L_pre_val, states_need))
        
        batch_counts, batch_L_pre, batch_states_need = self.batch_gathering(all_results,batch_size,mode=1)
        return torch.tensor(batch_counts), torch.tensor(batch_L_pre,device=get_device()), torch.tensor(batch_states_need,device=get_device())

    def batch_gathering(self,all_results,batch_size,mode=1):
        if mode==0:
            results_indices = np.concatenate([indices for indices,_,_ in all_results],axis=0)
            results_L_pre = np.concatenate([L_pre for _,L_pre,_ in all_results],axis=0)
            results_states_need = np.concatenate([states_need for _,_,states_need in all_results],axis=0)
            
            batch_counts = np.zeros(batch_size,dtype=np.int64)
            all_count =len(results_indices)
            batch_L_pre = np.zeros(all_count,dtype=np.complex128)
            batch_states_need = np.zeros((all_count,results_states_need.shape[1]),dtype=np.float64)

            for i in range(batch_size):
                condition = results_indices[:] == i
                i_indices = np.where(condition)[0]
                batch_counts_pre = batch_counts[i-1] if i>0 else 0
                batch_counts[i] = len(i_indices)+batch_counts_pre
                batch_L_pre[batch_counts_pre:batch_counts[i]] = results_L_pre[i_indices]
                batch_states_need[batch_counts_pre:batch_counts[i]] = results_states_need[i_indices]
        elif mode==1:
            batch_results = [[] for _ in range(batch_size)]
            for batch_indices, L_pre, states_need in all_results:
                for i in range(len(batch_indices)):
                    batch_results[batch_indices[i]].append((L_pre[i], states_need[i]))
            
            # Convert to final format
            batch_counts = np.array([len(results) for results in batch_results])
            batch_counts = np.cumsum(batch_counts)
            batch_L_pre = np.concatenate([np.array([L_pre for L_pre, _ in results], dtype=np.complex128) if results else np.array([], dtype=np.complex128) for results in batch_results])
            batch_states_need = np.concatenate([np.array([states_need for _, states_need in results], dtype=np.float64) if results else np.zeros((0, self.Ns*2+self.Nd), dtype=np.float64) for results in batch_results])
        return batch_counts, batch_L_pre, batch_states_need



    def Lforward(self,rho,states:np.ndarray,mode=None) :
        '''
        states:np.
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        return self.Lforward_batched(rho,states)

    def Gathering_L(self,count,L_pre,states_need,rho):
        n = count.shape[0]
        Lforward = torch.zeros(n,dtype=torch.complex128,device=get_device())
        rho_need = rho.States(states_need)
        Lforward_pre = torch.multiply(L_pre,rho_need)
        for i in range(0,n):
            if i == 0:
                Lforward[0] = torch.sum(Lforward_pre[0:count[0]])
            else:
                Lforward[i] = torch.sum(Lforward_pre[count[i-1]:count[i]])
        return Lforward
    
    
    def Lforward_batched(self,rho,states) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        count,L_pre,states_need = self.L_ls_batched(rho,states)
        return self.Gathering_L(count,L_pre,states_need,rho)


    def Lrho(self,rho,target,N = 20000) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        '''
        n = self.states.shape[0]
        L_forward = np.zeros(n,dtype=np.complex128)
        # L_forward_0 = torch.zeros(n,dtype=torch.complex128,device=self.device)
        count_pre = np.zeros(n,dtype=np.int32)
        count = 0
        j = 0
        L_pre = np.zeros(N,dtype=np.complex128)
        states_need = np.zeros((N,rho.nstate),dtype=np.float64)
        k_i = 0
        k_n = n
        for i in range(k_i,k_n):
            # if n%100 == 0:
            #     print(n,flush=True)
            # count_0, L_pre_0, states_need_0 = self.L_ls(rho,self.states[i])
            count_0, L_pre_0, states_need_0 = self.L_ls(rho,self.states[i])
            L_pre[count:count+count_0] = L_pre_0
            states_need[count:count+count_0] = states_need_0
            count_pre[i] = count_0
            count += count_0
            if count+rho.Nd*3 > N or i==k_n-1:
                print(f'count:{count}   i:{i}')
                # print(count)
                rho_states = np.zeros(count,dtype=np.complex128)
                for p in range(count):
                    if np.sum(states_need[p,:rho.Nd])<3.5:
                        index = np.argwhere(
                            np.sum(np.abs(self.states-states_need[p:p+1,:]),axis=1)<1e-6)
                        # print(f'index:{index}    L_pre:{L_pre[p]}')
                        # print(states_need[p])
                        if index.size==1:
                            rho_states[p] = target[index]
                    else:
                        rho_states[p] = 0.+0.j
                L_forward_pre = np.multiply(rho_states,L_pre[0:count])
                # L_forward_pre_0 = rho.States(states_need_torch)
                l = 0
                for k in range(j,i+1):
                    L_forward[k] = np.sum(L_forward_pre[l:l+count_pre[k]])
                    # L_forward_0[k] = torch.sum(L_forward_pre_0[l:l+count_pre[k]])
                    l = l+count_pre[k]
                count = 0
                j = i+1
        # print(f'L23:{L_forward}')
        print(f'L:{L_forward.dot(L_forward.conj())}')
        return L_forward
    
    
    def states_cut(self,rho,states,cut=2) :
        N = rho.Nd//2
        states = states
        state_cut = np.sum(states[:,0:rho.Nd],axis=1)
        ifcut = (-np.sign(state_cut-cut-0.1)+1)//2
        return (ifcut).to(np.complex128)


    def ifzero(self,x) :
        #x is a 1-d np_array
        if self.sys_mode == 0 :
            M_0 = np.zeros(self.Ns,dtype=x.dtype)
            M_1 = np.zeros(self.Ns,dtype=x.dtype)
            y = x[0:self.Nd].reshape(self.nsgn,self.Ns,-1)
            M_0 = np.sum(y[0,:,:],axis=1)
            M_1 = np.sum(y[1,:,:],axis=1)
            a = M_0-M_1+x[self.Nd:self.Nd+self.Ns]-x[self.Nd+self.Ns:self.nstate]
            if np.sum(a)<1e-6 :
                return True
            print(f'a: {a}')
            print(y)
            print(M_0)
            print(M_1)
            print(x[self.Nd:])
        return False


class LiouvilleSaveL(Liouville):
    def __init__(self,read,device,nonmccut,allcut,notsaving,mc_size=256,lmbda=-0.) :
        """
        This class save the L.
        """
        super().__init__(read,device,nonmccut,allcut,notsaving,mc_size,lmbda)

    def Lforward(self,rho,states=None,mode=0) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        if mode==0:
            Lforward = self.Lforward_prepared(rho)
        else:
            Lforward = self.Lforward_prepared_mc(rho)
        return Lforward
    
    def set_states_need(self,rho,states:np.ndarray,cut=3):
        '''
        calculate and save the states required to calculate Lrho
        '''
        t1 = time.time()
        count_pre, L_pre, states_need = self.Llocals(rho,states)
        t2 = time.time()
        print(f'time_nonmc-L_pre: {t2-t1:.2f}', flush=True)
        Llocals_pre = {
            'states':states,
            'count_pre':count_pre, 
            'L_pre':L_pre, 
            'states_need':states_need
            }
        self.states_need = Llocals_pre
        return 0
    
    def set_states_need_mc(self,rho,states_mc:np.ndarray,cut=3):
        '''
        calculate and save the states required to calculate Lrho using mc
        '''
        t1 = time.time()
        count_pre, L_pre, states_need = self.Llocals(rho,states_mc)
        t2 = time.time()
        print(f'time_mc-L_pre: {t2-t1:.2f}', flush=True)
        Llocals_pre = {
            'states_mc':states_mc,
            'count_pre_mc':count_pre, 
            'L_pre_mc':L_pre, 
            'states_need_mc':states_need
            }
        self.states_need_mc = Llocals_pre
        return 0

    def Lforward_prepared(self,rho) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        count = self.states_need['count_pre']
        L_pre = self.states_need['L_pre']
        states_need = self.states_need['states_need']
        return self.Gathering_L(count,L_pre,states_need,rho)
     
    def Lforward_prepared_mc(self,rho) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        count = self.states_need_mc['count_pre_mc']
        L_pre = self.states_need_mc['L_pre_mc']
        states_need = self.states_need_mc['states_need_mc']
        return self.Gathering_L(count,L_pre,states_need,rho)
    
    

class Liouvillet(Liouville):
    def __init__(self,read,device,nonmccut,allcut,notsaving=0,mc_size=0,lmbda=-0.) :
        """
        This class save the L.
        """
        super().__init__(read,device,nonmccut,allcut,notsaving=notsaving,mc_size=mc_size,lmbda=-0.)

    def Lforward(self,rho,states:np.ndarray,mode=None,t=0.) :
        '''
        states:np.
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        return self.Lforward_batched(rho,states,t)
    
    def Gathering_L(self,count,L_pre,states_need,rho,t):
        n = count.shape[0]
        Lforward = torch.zeros(n,dtype=torch.complex128,device=get_device())
        rho_need = rho.States(states_need,t)
        Lforward_pre = torch.multiply(L_pre,rho_need)
        for i in range(0,n):
            if i == 0:
                Lforward[0] = torch.sum(Lforward_pre[0:count[0]])
            else:
                Lforward[i] = torch.sum(Lforward_pre[count[i-1]:count[i]])
        return Lforward
    
    
    def Lforward_batched(self,rho,states:torch.Tensor,t) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        # print(states.dtype)
        states = states.to('cpu').numpy()
        count,L_pre,states_need = self.L_ls_batched(rho,states)
        return self.Gathering_L(count,L_pre,states_need,rho,t)
        
class LiouvilletSaveL(LiouvilleSaveL):
    def __init__(self,read,device,nonmccut,allcut,notsaving=0,mc_size=0,lmbda=-0.) :
        """
        This class save the L.
        """
        super().__init__(read,device,nonmccut,allcut,notsaving=notsaving,mc_size=mc_size,lmbda=-0.)

    def Lforward(self,rho,t,states=None,mode=0) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        if mode==0:
            Lforward = self.Lforward_prepared(rho,t)
        else:
            Lforward = self.Lforward_prepared_mc(rho,t)
        return Lforward
    
    def Gathering_L(self,count,L_pre,states_need,rho,t):
        n = count.shape[0]
        Lforward = torch.zeros(n,dtype=torch.complex128,device=get_device())
        rho_need = rho.States(states_need,t)
        Lforward_pre = torch.multiply(L_pre,rho_need)
        for i in range(0,n):
            if i == 0:
                Lforward[0] = torch.sum(Lforward_pre[0:count[0]])
            else:
                Lforward[i] = torch.sum(Lforward_pre[count[i-1]:count[i]])
        return Lforward
    
    def Lforward_prepared(self,rho,t) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        count = self.states_need['count_pre']
        L_pre = self.states_need['L_pre']
        states_need = self.states_need['states_need']
        return self.Gathering_L(count,L_pre,states_need,rho,t)
     
    def Lforward_prepared_mc(self,rho,t) :
        '''
        state should be two-dimensional (*,rho.nstate), * is the number of samples
        self.notsaving = 0, the required states and reltaed L_ij are saved in self.states_need 
        '''
        count = self.states_need_mc['count_pre_mc']
        L_pre = self.states_need_mc['L_pre_mc']
        states_need = self.states_need_mc['states_need_mc']
        return self.Gathering_L(count,L_pre,states_need,rho,t)