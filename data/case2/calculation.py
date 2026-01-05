import numpy as np
import scipy
import scipy.sparse
import scipy.linalg 
# import scipy.linalg 



listT = ['0.3','0.06','0.05','0.04','0.03']
listT_num = ['03','006','005','004','003']
# listT = ['0.3','0.2','0.1','0.06','0.05','0.04','0.03']
# listT_num = ['03','02','01','006','005','004','003']
nrho = 16



# quick_rho0 = {}
# quick_SvN = {}
# for i in range(len(listT_num)):
#     with open("quick_rho0_T="+listT[i]) as f: 
#         quick_rho0['T='+listT[i]] = np.array(f.read().split(), dtype=float).reshape(200, 273)
#         quick_SvN['T='+listT[i]] = np.zeros((200,2),dtype=np.float64)
#     for j in range(200):
#         rho0 = np.zeros((nrho,nrho),dtype=np.complex128)
#         for k in range(nrho):
#             rho0[k,0:k+1] = quick_rho0['T='+listT[i]][j,(1+(k*(k+1))//2):(1+((k+1)*(k+2))//2)]+1.j*quick_rho0['T='+listT[i]][j,(137+(k*(k+1))//2):(137+((k+1)*(k+2))//2)]
#         rho0 = rho0 + np.triu(rho0.T.conj(),k=1)
#         quick_SvN['T='+listT[i]][j,1] = -np.real(np.trace(np.matmul(rho0,scipy.linalg.logm(rho0))))
#     quick_SvN['T='+listT[i]][:,0] = quick_rho0['T='+listT[i]][:,0]

#     f = open('quick_SvN_T='+listT[i],'w')
#     for j in range(200):
#         tt = quick_SvN[f'T='+listT[i]][j,0]
#         result = quick_SvN[f'T='+listT[i]][j,1]
#         f.write(f'      {tt:.6f}   {result: .6e}\n')
#         f.flush()
#     f.close()



listT = ['0.3']
listT_num = ['03']


Hsys = np.loadtxt("ham_sys.data", skiprows=1, dtype=np.complex128).reshape(nrho,nrho)
list_rho0 = [j for j in range(1,nrho+5)]
listT_rho0 = {}
vN_entropy, Esys_T = {}, {}
for i in range(len(listT_num)):
    # listT_rho0['T='+listT[i]] = np.loadtxt("rbm_rho0_elem_T="+listT[i], dtype=np.float64)[0:]
    listT_rho0['T='+listT[i]] = np.loadtxt("rbm_rho0_elem_T="+listT[i]+'_mc', dtype=np.float64)[0:]
    lenT_rho0 = len(listT_rho0['T='+listT[i]][:,0])
    Esys_T['T='+listT[i]] = np.zeros((lenT_rho0,2),dtype=np.float64)
    vN_entropy['T='+listT[i]] = np.zeros((lenT_rho0,2),dtype=np.float64)
    for j in range(lenT_rho0):
        rho0 = np.zeros((nrho,nrho),dtype=np.complex128)
        rho0_elem = listT_rho0['T='+listT[i]][j,list_rho0]
        for k in range(nrho):
            rho0[k][k] = rho0_elem[k]
        rho0[6][9] = rho0_elem[nrho] + 1.j*rho0_elem[nrho+2]
        rho0[9][6] = rho0_elem[nrho+1] + 1.j*rho0_elem[nrho+3]
        Esys_T['T='+listT[i]][j,1] = np.real(np.trace(np.matmul(Hsys,rho0)))
        vN_entropy['T='+listT[i]][j,1] = -np.real(np.trace(np.matmul(rho0,scipy.linalg.logm(rho0))))
    Esys_T['T='+listT[i]][:,0] = listT_rho0['T='+listT[i]][:,0]
    vN_entropy['T='+listT[i]][:,0] = listT_rho0['T='+listT[i]][:,0]

    # f1 = open('rbm_SvN_T='+listT[i],'w')
    f1 = open('rbm_SvN_T='+listT[i]+'_mc','w')
    for j in range(lenT_rho0):
        tt = vN_entropy[f'T='+listT[i]][j,0]
        result = vN_entropy[f'T='+listT[i]][j,1]
        f1.write(f'      {tt:.6f}   {result: .6e}\n')
        f1.flush()
    f1.close()

    # f2 = open('rbm_Esys_T='+listT[i],'w')
    f2 = open('rbm_Esys_T='+listT[i]+'_mc','w')
    for j in range(lenT_rho0):
        tt = Esys_T[f'T='+listT[i]][j,0]
        result = Esys_T[f'T='+listT[i]][j,1]
        f2.write(f'      {tt:.6f}   {result: .6e}\n')
        f2.flush()
    f2.close()

































# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq
# from scipy.signal import find_peaks
# listT = ['0.3','0.06','0.05','0.04','0.03']
# listT_num = ['03','006','005','004','003']

# rbm_tstep, quick_ddot = {}, {}
# rbm_Esys, quick_energy = {}, {}
# rbm_SvN, quick_SvN = {}, {}
# for i in range(len(listT_num)):
#     # rbm_tstep['T='+listT[i]] = np.loadtxt("rbm_t-step_T="+listT[i], usecols=(0,9), dtype=np.float64)
#     quick_ddot['T='+listT[i]] = np.loadtxt("quick_ddot_T="+listT[i], usecols=(0,1), dtype=np.float64)
#     # rbm_Esys['T='+listT[i]] = np.loadtxt("rbm_Esys_T="+listT[i], dtype=np.float64)
#     quick_energy['T='+listT[i]] = np.loadtxt("quick_engy_T="+listT[i], dtype=np.float64)
#     # rbm_SvN['T='+listT[i]] = np.loadtxt("rbm_SvN_T="+listT[i], dtype=np.float64)
#     quick_SvN['T='+listT[i]] = np.loadtxt("quick_SvN_T="+listT[i], dtype=np.float64)

    
# i=4
# f = quick_SvN['T='+listT[i]]
# diff = np.diff(f[:,1])/np.diff(f[:,0])
# diff_x = (f[:,0][:-1]+f[:,0][1:])/2
# diff2 = np.diff(diff)/np.diff(diff_x)
# diff2_x = (diff_x[:-1]+diff_x[1:])/2


# # print(scipy.signal.find_peaks(diff))

# y = diff2
# sumf = fft(y)

# N = y.shape[0]
# delta_T = f[1,0]-f[0,0]
# print(delta_T)
# # xf = fftfreq(N, delta_T)[:N//2]
# x = [i for i in range(0,N//2)]
# x = np.array(x)
# x_ = 1/(x+1e-6)*N*delta_T
# yf = (2.0/N*np.abs(sumf[0:N//2]))
# print(sumf.shape,N,x.shape,yf.shape)
# plt.figure('Fourier')
# # plt.xscale("log")
# plt.plot(x_, yf)
# plt.xlim(1,12)
# plt.savefig('Fourier.png', dpi=100)


# # diff = np.diff(yf)/np.diff(x)
# # diff_x = (x[:-1])
# # peaks, _ = find_peaks(diff[:100])
# # print(peaks)
# # print(diff_x[peaks[0]])
# # print(diff[peaks[0]])
# # # plt.plot(diff_x, diff)


# peaks, _ = find_peaks(yf[:100])
# print(peaks)
# print(x_[peaks[0]],x[peaks[0]])
