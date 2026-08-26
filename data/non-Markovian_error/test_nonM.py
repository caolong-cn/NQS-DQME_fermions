import torch
import numpy as np
import sys
import time
import os
from scipy.sparse.linalg import eigs

sys.path.insert(0, '/home/caolong/NQS-DQME_1/src')

from nqsdqme.state import DecoupledState,ExactStateSave
from nqsdqme.core.sysham import SysHam
from nqsdqme.core.dissipation import Dissipation
from nqsdqme.core.liouville import Liouville,LiouvilleSaveL
from nqsdqme.core.read import ReadHamilton
from nqsdqme.core.operators import Operators

from nqsdqme.core.basis import Basis

from nqsdqme.global_defs import update_device,get_device


from scipy.sparse import csr_matrix, lil_matrix
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv

from scipy.sparse.linalg import spsolve, expm_multiply
from scipy.sparse.linalg import factorized

import scipy.sparse as sp


def setup_converters(table):
    """
    预处理 table 以生成快速转换函数
    """
    # 1. 向量化反向查找：从 态s 得到 索引 (convert)
    # 对于一维态，通常将其视为一行。我们利用字典或 NumPy 的 searchsorted（如果已排序）
    # 这里使用字典映射，因为它对任意状态分布都非常快
    state_to_idx = {tuple(state): i for i, state in enumerate(table)}
    

    def convert(states):
        """
        将多个态组成的矩阵转换为索引向量
        输入 states: (M, D) 矩阵
        输出: (M,) 索引数组
        """
        # 利用 map 快速处理，比 Python 原生循环快得多
        # indexs = np.zeros(states.shape[0],dtype=np.int64)
        # for i in range(states.shape[0]):
        #     delta = np.sum(np.abs(states[i]-table),axis=1)
        #     a = np.argwhere(
        #         delta<1e-6)
        #     # np.where(delta[:]==0)[0]
        #     if len(a)==0:
        #         print(table.shape)
        #         print(delta)
        #         print(states[i])
        #         print(np.sum(states[i,0:rho.Nd]))
        #     indexs[i] = a[0]
        # return indexs
        return np.array([state_to_idx[tuple(map(int,s))] for s in states])

        # return np.array([state_to_idx[tuple(s)] for s in states])

    # 2. 向量化正向查找：从 索引 得到 态s (iconvert)
    def iconvert(indices):
        """
        将索引向量转换为对应的态矩阵
        输入 indices: (M,) 索引数组
        输出: (M, D) 态矩阵
        """
        # NumPy 的高级索引本身就是天然向量化的
        return table[indices]

    return convert, iconvert

def setup_converters_m(table):
    mapping = {}

    for idx, row in enumerate(table):
        # Convert the first M elements to a tuple so it can be a dict key
        prefix = tuple(row)
        
        if prefix not in mapping:
            mapping[prefix] = []
        mapping[prefix].append(idx)

    # def convert_M(s):
    #     return np.array(mapping[tuple(map(int,s))])
    return mapping
        
def indexes_M(table,M):
    """
    返回 table 中满足dissipaton占据数等于 M的行索引。

    参数：
        table : np.ndarray, shape (n_rows, n_cols)
            二维数组，每一行代表一个态（state），列代表该态的各个分量或量子数。
        M : int
            目标求和值。

    返回：
        indexes : np.ndarray
            满足条件的行索引数组。若没有符合条件的行，则返回空数组。
    """
    indexes = np.where(np.sum(table[:,0:liouville.Nd],axis=1)==M)[0]

    return indexes


def build_explicit_L(all_states, L_ls_batch, convert):
    """
    all_states: 包含所有层级矢量 (s) 的列表
    L_ls: 函数, 输入 s, 返回 (nonzero_s_prime, matrix_elements)
    convert: 函数, 将矢量 s 转换为矩阵索引 i
    """
    hbar = 0.658211928 #eV*fs
    dim = len(all_states)
    print(dim)
    L_sparse = lil_matrix((dim, dim), dtype=complex)
    # L_sparse[idx_row, idx_col] += val
    batch_counts, batch_L_pre, batch_states_need = L_ls_batch(all_states)
    if isinstance(batch_counts,torch.Tensor):
        batch_counts = batch_counts.to('cpu').numpy()
        batch_L_pre = batch_L_pre.to('cpu').numpy()
        batch_states_need = batch_states_need.to('cpu').numpy().astype(np.int8)
    # batch_states_need = map(int, batch_states_need)
    # print(batch_states_need)
    # print(all_states)
    # print(batch_states_need)
    all_s = convert(all_states)
    # print(all_s)
    all_s_prime = convert(batch_states_need)
    for i in range(all_states.shape[0]):
        idx_row = all_s[i]
        # 获取 L 矩阵中该行对应的所有非零元
        if i == 0:
            next_states = all_s_prime[0:batch_counts[0]]
            values = batch_L_pre[0:batch_counts[0]]
        else:
            next_states = all_s_prime[batch_counts[i-1]:batch_counts[i]]
            values = batch_L_pre[batch_counts[i-1]:batch_counts[i]]

        for s_prime, val in zip(next_states, values):
            idx_col = s_prime
            L_sparse[idx_row, idx_col] += val          
    return (L_sparse*hbar/unit_E).tocsr()  #convert to the eV


def calculate_chi_trajectory(L_matrix, t_array, mapping,n_physics=4,M_max=3 , eps=1e-12):
    """
    L_matrix: 显式构造的 NumPy 方阵
    t_array: 时间点序列
    """
    dim = L_matrix.shape[0]
    
    # 1. 正则化处理 L 的奇异性 (稳态零特征值)
    # 采用 L - eps*I 确保可逆，这在物理上相当于引入了一个极微弱的全局衰减
    # L_reg = L_matrix - eps * np.eye(dim)
    # L_inv = inv(L_reg)
    
    # 存储结果：每一行对应一个时间点，每一列对应一个dissipaton占据数M
    chi_t = np.zeros((len(t_array), M_max+1))
    
    # 2. 计算轨迹
    # 为了提高效率，如果 t_array 很长，建议使用特征对角化加速，
    # 这里为了通用性使用 expm
    idxs_phys = np.arange(n_physics)
    I = np.eye(dim)
    
    for i, t in enumerate(t_array):
        M_t = get_top_rows_M(L_matrix,t,n_physics,eps)
        # 计算解析解矩阵 M(t)
        # M_t = (L_inv @ (expm(L_matrix * t) - I))[0:n_physics,:]
        
        # 提取物理层对所有 m 的响应强度
        # 我们取模长（Frobenius 范数在标量位上即绝对值）
        for key, val in mapping.items():
            # print(key)
            M = int(np.sum(np.array(key)))
            # print(M)
            cols = np.array(val)
            # print(len(val))
            Frobenius_m = np.linalg.norm(M_t[:,cols], 'fro') 
            chi_t[i,M] += Frobenius_m
        print(f't:{t_array[i]}  chi_t:{chi_t[i,:]}',flush=True)
    return chi_t

def calculate_chi_distribution(L_matrix, t, f_p,mapping,n_physics=4,M_max=3 , eps=1e-12):
    """
    L_matrix: 显式构造的 NumPy 方阵
    t_array: 时间点序列
    """
    dim = L_matrix.shape[0]
    ncor = dissipation.ncor
    dissipation.init_jm()
    # 1. 正则化处理 L 的奇异性 (稳态零特征值)
    # 采用 L - eps*I 确保可逆，这在物理上相当于引入了一个极微弱的全局衰减
    # L_reg = L_matrix - eps * np.eye(dim)
    # L_inv = inv(L_reg)
    
    # 存储结果：每一行对应一个occupation number in the corresponding mode
    # ，每一列对应一个dissipaton mode m
    chi_Mm = np.zeros((M_max, ncor))
    
    # 2. 计算轨迹
    # 为了提高效率，如果 t_array 很长，建议使用特征对角化加速，
    # 这里为了通用性使用 expm
    idxs_phys = np.arange(n_physics)
    I = np.eye(dim)
    

    M_t = get_top_rows_M(L_matrix,t,n_physics,eps)
    # 计算解析解矩阵 M(t)
    # M_t = (L_inv @ (expm(L_matrix * t) - I))[0:n_physics,:]
    
    # 提取物理层对所有 m 的响应强度
    # 我们取模长（Frobenius 范数在标量位上即绝对值）
    for key, val in mapping.items():
        # print(key)
        m = np.array(key)
        M_i = int(np.sum(m))
        # if M==M_i:
        # print(M)
        cols = np.array(val)
        # print(len(val))
        Frobenius_m = np.linalg.norm(M_t[:,cols], 'fro') 
        nonzerojs = np.nonzero(m)[0]
        ms = np.zeros_like(nonzerojs)
        for i in range(len(nonzerojs)):
            ms[i] = dissipation.j_to_m[nonzerojs[i]]
        unique, counts = np.unique(ms, return_counts=True)
        weight_p = counts*f_p[unique]
        weight_p = weight_p/np.sum(weight_p)
        for i in range(len(unique)):
            chi_Mm[counts[i]-1,unique[i]] += Frobenius_m*weight_p[i]

    print(f't:{t}   \n',flush=True)
    # print(f'the distribution of chi in different modes:\n {chi_Mm}')
    chi_Mm = np.sum(chi_Mm,axis=0)
    print(f'the distribution of chi in different modes:\n {chi_Mm}')
    return chi_Mm

def calculate_chi_distribution_perlayer(L_matrix, t, mapping,n_physics=4,M=1 , eps=1e-12):
    """
    L_matrix: 显式构造的 NumPy 方阵
    t_array: 时间点序列
    """
    dim = L_matrix.shape[0]
    ncor = dissipation.ncor
    dissipation.init_jm()
    # 1. 正则化处理 L 的奇异性 (稳态零特征值)
    # 采用 L - eps*I 确保可逆，这在物理上相当于引入了一个极微弱的全局衰减
    # L_reg = L_matrix - eps * np.eye(dim)
    # L_inv = inv(L_reg)
    
    # 存储结果：每一行对应一个occupation number in the corresponding mode
    # ，每一列对应一个dissipaton mode m
    chi_Mm = np.zeros((M, ncor))
    
    # 2. 计算轨迹
    # 为了提高效率，如果 t_array 很长，建议使用特征对角化加速，
    # 这里为了通用性使用 expm
    idxs_phys = np.arange(n_physics)
    I = np.eye(dim)
    

    M_t = get_top_rows_M(L_matrix,t,n_physics,eps)
    # 计算解析解矩阵 M(t)
    # M_t = (L_inv @ (expm(L_matrix * t) - I))[0:n_physics,:]
    
    # 提取物理层对所有 m 的响应强度
    # 我们取模长（Frobenius 范数在标量位上即绝对值）
    for key, val in mapping.items():
        # print(key)
        m = np.array(key)
        M_i = int(np.sum(m))
        if M_i==M:
        # print(M)
            cols = np.array(val)
            # print(len(val))
            Frobenius_m = np.linalg.norm(M_t[:,cols], 'fro') 
            nonzerojs = np.nonzero(m)[0]
            ms = np.zeros_like(nonzerojs)
            for i in range(len(nonzerojs)):
                ms[i] = dissipation.j_to_m[nonzerojs[i]]
            unique, counts = np.unique(ms, return_counts=True)
            total_counts = np.sum(counts)
            for i in range(len(unique)):
                chi_Mm[counts[i]-1,unique[i]] += Frobenius_m*counts[i]/total_counts
            # print(total_counts)
    print(f't:{t}   \n',flush=True)
    # print(f'the distribution of chi in different modes:\n {chi_Mm}')
    chi_Mm = np.sum(chi_Mm,axis=0)
    print(f'the distribution of chi in different modes cut 1:\n {chi_Mm}')
    return chi_Mm

def get_top_rows_M(L, t, r=4,eps=1.e-12):
    # eps = 0.
    n = L.shape[0]
    # L = csr_matrix(L)
    if isinstance(L,np.ndarray):
        L = csr_matrix(L)
    LT_eps = (L- eps * sp.eye(n, format='csr')).T.tocsr()  
    if  regularization_mode==0:
        LT = L.T.tocsr()    # 转置, only regularize the denominator
    else:
        LT =  LT_eps         
    solve_LT = factorized(LT_eps)            # 预分解，用于快速求解 L^T v = e_i
    rows = []
    for i in range(r):
        e_i = np.zeros(n, dtype=np.complex128)
        e_i[i] = 1.0
        v_i = solve_LT(e_i)              # 求解 L^T v_i = e_i
        w_i = expm_multiply(LT * t, v_i) # 计算 expm(L^T t) @ v_i
        row_i = w_i - v_i                # X 的第 i 行
        rows.append(row_i)
    return np.array(rows)                # shape (r, n)


color = ['black','#2C5AB6','#428EA8','#6B2876','#9A759F','#c82d31']
def plot_chi_evolution(chi_t, t_array):
    for i in range(chi_t.shape[1]):
        plt.plot(t_array, chi_t[:,i], label=f'Tier {i}', linewidth=1, color=color[i])
    
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Cumulative Sensitivity $\chi(t)$', fontsize=12)
    plt.title('Error Propagation: Tier-wise Sensitivity to Residuals', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    if chi_t[-1,-1]/chi_t[-1,0] > 100:
        plt.yscale('log') # 强非马尔可夫区建议开启对数轴
    plt.savefig('case1_hight_chi'+str(dissipation.Ts[0])+'.png',dpi=100)
    return 0

delta = (0.1e0 + 0.1e0) / 2  #eV
unit_E = delta
unit_t = 1/delta  #eV ^-1

# L_dense = L_matrix.toarray()

#input文件
#cuda/cpu
#hidden auxillary 0
#nonmccut allcut 0(>nonmccut <=allcut的做mc)  (nonmccut与allcut按照Nrho算)
#chains Nstart Nend
#t0 count

#initialize hamilton and device
input_device = np.loadtxt("input",max_rows=1,dtype=np.str_)
device = torch.device(str(input_device))
update_device(device)
print(f'device:{device}')


#initialize hyperparameters and define the problem to be solved  
input_para = np.loadtxt("input",skiprows=1,max_rows=3,dtype=np.int32)
input_t = np.loadtxt("input",skiprows=4,max_rows=1,dtype=np.float64)
nhidden = input_para[0,0]
nauxillary = input_para[0,1]
epsilonmode = input_para[0,2]
nonmccut = 3
allcut = nonmccut
notsaving = input_para[1,2] # = 0(1): (not) saving L
# chains_number = input_para[2,0]
# N_start = input_para[2,1]
# N_end = input_para[2,2]
mc_size = input_para[2,0]

ham_para,env_para,n_para = ReadHamilton()

sys_para = ham_para | n_para
sys_para['allcut'] = allcut
sysham = SysHam(sys_para,get_device())


dissipation_para = env_para | n_para
dissipation_para['allcut'] = allcut
dissipation_para['sys_mode'] = sys_para['sys_mode']
dissipation = Dissipation(dissipation_para,get_device())

# liouville = LiouvilleSaveL(sysham,dissipation,nonmccut,notsaving,mc_size=mc_size)
# if nonmccut<allcut:
#     liouville.lmbda = -3.



# coupling_map = np.zeros((sysham.nvar,dissipation_para['nalf']),dtype=np.int64)
# coupling_map[0,0] = 1
# coupling_map[-1,-1] = 1
# dissipation_para['coupling_map'] = coupling_map
# sys.exit()
liouville = Liouville(sysham,dissipation,nonmccut,notsaving,mc_size=mc_size)

#initialize the basis and sampler
basis = Basis(liouville,nonmccut,allcut,ifread=False)
table1 = basis.table(1)
print(f'Ns:{basis.Ns}')

operators = Operators(liouville,basis.states)

# print(f'counts of <= {allcut} tier ado elements: {basis.get_total_tier_count(allcut+1)}')
    
# rho = DecoupledState(liouville.Nd,liouville.Ns, basis)

M_max = allcut
fname = "table_cut"+str(M_max+1)+".data"
table = basis.states
print(table.shape)
print(f"Nd: {liouville.Nd}")
# hbar = 0.658211928
# print(sysham.Hs_csr*hbar)
# np.loadtxt(fname,usecols=lambda i: i >= 3,dtype=np.int8)



convert,iconvert = setup_converters(table)

mapping = setup_converters_m(table[:,:liouville.Nd])


L_matrix = build_explicit_L(table,liouville.L_ls_batched,convert)

# L_matrix = build_explicit_L(table,dissipation.D_ls_batched,convert)
# k: number of eigenvalues to find
# which='LR': Largest Real part (for decay modes)
# which='SM': Smallest Magnitude (for the steady state)
k = 4
eigenvalues, eigenvectors = eigs(L_matrix, k=k, which='LR')

print(f"Top {k} Eigenvalues:    {eigenvalues}")

# for eig_idx in range(k):
#     rho = ExactStateSave(basis.Nd, basis.Ns, basis, device, table, eigenvectors[:,eig_idx])
#     rho_0 = rho.rho_0().detach().cpu().numpy()
#     rho_0 = np.real(rho_0/np.trace(rho_0))
#     print(f"the occupation number of the {eig_idx} Eigenstate:    ")
#     for i in range(liouville.nvar):
#         n_up, n_down = operators.occupation(rho_0,i)
#         print(f'{i}:  {n_up:.5e}  {n_down:.5e}   \n')

#     trace = np.real(np.trace(rho_0))
#     print(f'trace:  {trace} \n')

# nstate = 2*liouville.nvar*liouville.nspin + liouville.nsgn*liouville.nvar*liouville.nspin*liouville.ncor*liouville.nalf
# Ns = liouville.nvar * liouville.nspin
# Nd = liouville.nsgn * Ns * liouville.nalf * liouville.ncor
# statescols = [i for i in range(3,3+nstate)]
# filename="table_cut"+str(nonmccut+1)+".data"
# table = np.loadtxt(filename,usecols=(statescols),dtype=np.int8)
# values = np.loadtxt(filename,usecols=(0,1),dtype=np.float64)
# values = values[:,0] + 1.j *values[:,1]

# rho = ExactStateSave(Nd, Ns, basis, device, table, values)

# print(f'LdaggerL:    {liouville.LdaggerL(rho,basis.states)}')
# eigv = rho.States(basis.states).cpu().numpy()
# res = L_matrix @ eigv
# print(f'LdaggerL:    {np.conj(res.T) @ res}')


# deltav = values[convert(table)]-values
# print(deltav)

# [6.,8.,10.,12.,20.]
t_array = np.array([6.,8.,10.,12.,20.])
# np.arange(0,15,1.)
# t_array = np.array([0.,0.1,0.2,0.4,0.6,0.8,1.0,1.25,1.5,2.,2.5,3.,3.5,4.])
# t_array = np.array([0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,1.0,1.25,1.5,2.,2.5,3.,3.5,4.])
# np.arange(0,500,50.)
# np.array([0.,0.1,0.2,0.4,0.6,0.8,1.0,1.25,1.5,2.,2.5,3.,3.5,4.])

regularization_mode = 0

eps = 1.e-12
print(f'T:{liouville.Ts[0]}')
print(t_array)
print(f'eps:{eps}')
print(f'regularization_mode:{regularization_mode}')
L_dense = L_matrix.toarray()
# print(np.sum(np.abs(L_dense)))
# print(np.sum(L_dense))

# sys.exit()

indexes_Ms = []
for i in range(M_max+1):
    indexes_Ms.append(indexes_M(table,i))

# print(indexes_Ms[0])
n_physics = len(indexes_Ms[0])



# f_p = calculate_chi_distribution_perlayer(L_dense,20.,mapping,n_physics,M=1,eps=eps)
# f_p = f_p/np.sum(f_p)

# calculate_chi_distribution(L_dense,20.,f_p,mapping,n_physics,M_max=M_max,eps=eps)

# chi_t = calculate_chi_trajectory(L_dense,t_array,mapping,n_physics,M_max=M_max,eps=eps)

# plot_chi_evolution(chi_t,t_array)









# sys.exit()

t1 = time.time()
chi_t = calculate_chi_trajectory(L_dense,t_array,mapping,n_physics,M_max=M_max,eps=eps)
t2 = time.time()
print(f"time_consumed:   {t2-t1}")
plot_chi_evolution(chi_t,t_array)

with open('input_env', 'r') as f:
    line = f.readline()
parts = line.split(':')
Ts = parts[1].split('#')[0].strip()
result_chi = np.column_stack((t_array,chi_t))
# np.savetxt("T_"+Ts+"_m3_Nb1_Ns6",result_chi)
np.savetxt("T_"+Ts+"_6-12",result_chi)

