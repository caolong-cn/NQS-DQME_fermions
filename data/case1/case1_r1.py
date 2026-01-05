import numpy as np
import scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys
import math
from scipy.optimize import curve_fit

from derivative import integral_error,power_law_fit
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset




# import shutil
# import matplotlib
# from matplotlib import font_manager  
# shutil.rmtree(matplotlib.get_cachedir())
# # for font in font_manager.fontManager.ttflist:  
# #     print(font.name, '-', font.fname)


# fname = 'Latin Modern Roman'
# fname_italic = 'Latin Modern Roman:italic'
# fname = 'Cambria'
# fname_italic = 'Cambria:italic'
# fname2 = 'Arial'
# plt.rcParams['text.usetex'] = True
class latexmath:
    def __enter__(self):
        plt.rcParams['mathtext.fontset'] = 'cm'
        return 0
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        plt.rcParams['mathtext.fontset'] = 'custom'
        return 0

fname = 'Arial'
fname_italic = 'Arial:italic'
fname2 = 'Arial'

plt.rcParams['font.family'] = fname  #使得坐标轴刻度标签字体变化
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = fname
plt.rcParams['mathtext.it'] = fname_italic
# plt.rcParams['mathtext.fontset'] = 'cm'
time_integral = 25



# unit
delta = (0.1e0 + 0.1e0) / 2  #meV
listT_unit = ['3.0','2.0','1.0','0.6','0.3']
meV = 1.6021766208e-19*1e-3   #J
h = 6.62607015e-34      #J·s
e = 1.602176634e-19     #C
pi = 3.1415926535897932
unit_t = 1e-12 * (delta * meV) / h *2*pi   #hbar/\gamma
unit_I = 1e-12 * h / (e * delta * meV) /(2*pi)   #e\gamma/hbar
# print(unit_t)
# sys.exit()
#general layout
fig = plt.figure(figsize=(8.6,10.3))

ax_a = fig.add_axes([0.095,0.693,0.855,0.25])
ax_a.axis('off')
ax_a.text(-0.04,1.00,r'${\bf (a)}$',fontdict={'family':fname2,'size': 16},transform=ax_a.transAxes)

# ax_c = fig.add_axes([0.09,0.40,0.87,0.265])
# ax_b.text(-0.04,1.05,'(b)',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)
ax_b = fig.add_axes([0.095,0.410,0.39,0.265])
ax_b.text(-0.08,1.05,r'${\bf (b)}$',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)

ax_c = fig.add_axes([0.58,0.410,0.4,0.265])
ax_c.text(-0.08,1.05,r'${\bf (c)}$',fontdict={'family':fname2,'size': 16},transform=ax_c.transAxes)

ax_d = fig.add_axes([0.1,0.056,0.41,0.265])
ax_d.text(-0.08,1.05,r'${\bf (d)}$',fontdict={'family':fname2,'size': 16},transform=ax_d.transAxes)

ax_e = fig.add_axes([0.585,0.056,0.38,0.265])
ax_e.text(-0.08,1.05,r'${\bf (e)}$',fontdict={'family':fname2,'size': 16},transform=ax_e.transAxes)

inset_b = fig.add_axes([0.238,0.565,0.165,0.094])

inset_c = fig.add_axes([0.82,0.565,0.15,0.1])

inset_d = fig.add_axes([0.19,0.2,0.135,0.095])




#case1示意图
# case1 = plt.imread('case1_a.png')
# ax_a.imshow(case1)#,cmap='hot'
# ax_a.set_xticks(())
# ax_a.set_yticks(())

#Temp
listT = ['0.3','0.2','0.1','0.06','0.03']    #0.03 using t-step_mc
listT_num = ['03','02','01','006','003']
rbm, quick = {}, {}
for i in range(len(listT_num)):
    rbm['T='+listT[i]] = np.loadtxt("t-step_"+listT[i], dtype=np.float64)
    quick['T='+listT[i]] = np.loadtxt("quick_"+listT[i], dtype=np.float64)
    # rbm['T='+listT[i]] = rbm['T='+listT[i]][rbm['T='+listT[i]][:,0]<25]
    # quick['T='+listT[i]] = quick['T='+listT[i]][quick['T='+listT[i]][:,0]<25]
I_long = np.array([5.71234e+02, 6.07858e+02, 6.37003e+02, 7.54458e+02, 1.08598e+03])  #I_Right
I_quick = np.array([5.71050E+002, 0.608078E+003, 0.635714E+003, 7.52483E+002, 1.09214E+003])  #I_Right
n_all_long = np.array([5.07137e-01+5.07118e-01,5.06597e-01+5.06595e-01,5.05110e-01+5.05175e-01,5.04734e-01+5.04807e-01,5.03723e-01+5.05172e-01])
n_up_long = np.array([5.07137e-01, 5.06597e-01, 5.05110e-01, 5.04734e-01, 5.03723e-01])
T = np.array([0.3, 0.2, 0.1, 0.06, 0.03])

# color = ['black','#c82d31','#194f97','#bd6b08','#00994e']#black, red, blue, yellow, green
# color = ['black','#c82d31','#2C5AB6','#428EA8','#5C2366','#9E7CA3']#黑, 红, 蓝, 青, 深紫, 淡紫, 
color = ['black','#2C5AB6','#428EA8','#6B2876','#9A759F','#c82d31']#黑, 蓝, 青, 深紫, 淡紫, 红


ax_b.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 18})
ax_b.set_ylabel(r'$I_{\rm R}$ ($\Gamma$)',fontdict={'family':fname2,'size': 17})
for i in range(len(listT_num)):
   ax_b.plot(rbm['T='+listT[i]][:,0]*unit_t,(rbm['T='+listT[i]][:,7]-I_long[i]+4600*i)*unit_I,linewidth='1.2',color=color[i+1],zorder=i+len(listT_num))
   ax_b.plot(quick['T='+listT[i]][:,0]*unit_t,(quick['T='+listT[i]][:,4]-I_long[i]+4600*i)*unit_I,linewidth='1.0',color=color[i+1],linestyle='--',zorder=i)
for i in range(len(listT_num)):
    ax_b.text(16*unit_t,(4600*i+800)*unit_I,s=r'$k_{\rm B}T=$'+listT_unit[i]+r'$\Gamma$',fontsize=15,color=color[i+1],ha='left',va='baseline')
# print('I_R:')
# for i in range(len(listT_num)):
#     scale0 = rbm['T='+listT[i]][:,0]<= 25
#     # time_integral
#     rbm_x,rbm_y = rbm['T='+listT[i]][:,0][scale0],rbm['T='+listT[i]][:,7][scale0]
#     scale1 = quick['T='+listT[i]][:,0]<=rbm_x[-1]
#     ref_x,ref_y = quick['T='+listT[i]][:,0][scale1],quick['T='+listT[i]][:,4][scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm_y)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
ax_b.plot([],[],label='RBM',linewidth='1.2',color=color[0])
ax_b.plot([],[],label='ref.',linewidth='1.2',color=color[0],linestyle='--')
# ax_b.legend(loc=1,prop={'size':16},ncol=1,frameon=False)
# ax_b.legend(bbox_to_anchor=(1.04,1.01),loc=1,prop={'size':16},handletextpad=0.6,handlelength=2,frameon=False)



T_j_cut4 = np.loadtxt("T_j_cut4", usecols=(2), dtype = np.float64)
T_inverse_cut4 = np.array([(1/1.0+i*(1/0.025-1/1.0)/(len(T_j_cut4)-1)) for i in range(len(T_j_cut4))])
# T_inverse_cut4 = [(1/0.3+i*(1/0.03-1/0.3)/(len(T_j_cut4)-1)) for i in range(len(T_j_cut4))]
inset_b.set_xlabel(r'$\Gamma/k_{\rm B}T$',fontsize=13,labelpad=2)
inset_b.set_ylabel(r'$I_{\rm R}(t_{\rm lo})(\Gamma)$',fontsize=14,labelpad=2)
# inset_b.set_ylabel(r'$I$',fontsize=12,labelpad=2)
# inset_b.plot(1/T,I_long,linewidth='1',color=color[0])
# inset_b.scatter(1/T,I_long,s=22,marker='^',color=color[-1],zorder=2)
inset_b.plot((1/T)*delta,I_long*unit_I,linewidth='0',marker='^',ms=4.5,color=color[-1],zorder=2) # ms=4.5
# inset_b.plot((1/T)*delta,I_quick*unit_I,linewidth='0',marker='o',ms=4.5,color=color[0],zorder=1) # ms=4.5
inset_b.plot(delta*T_inverse_cut4[1:-3],T_j_cut4[1:-3]*unit_I,linewidth='1.0',linestyle='-',color=color[0],zorder=1)
# ,linestyle='--'
inset_b.plot([],[],label='RBM',linewidth='0',marker='^',ms=4.5,color=color[-1])
inset_b.plot([],[],label='ref.',linestyle='-',linewidth='1.0',ms=4.5,color=color[0])
# inset_b.plot([],[],label='ref.',linewidth='0',marker='o',ms=4.5,color=color[0])
inset_b.legend(bbox_to_anchor=(-0.03,1.05),loc=2,prop={'size':11.5},handletextpad=0.5,handlelength=1.5,frameon=False)
# inset_b.legend(bbox_to_anchor=(-0.00,1.05),loc=2,prop={'size':10},handlelength=0.2,frameon=False)







#Npara

def plot_Nh(ax:Axes,inset:Axes,j=7):

    if j==7:
        j_q = 4
    ref = quick['T=0.03']
    ref = ref[ref[:,0]<25]
    ax.set_yscale("log")
    ax.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 17})
    ax.set_ylabel(r'$\Delta {s}^{2}$',fontsize=16)
    ax.tick_params(axis='y',labelsize=12,length=4,width=0.8)
    ax.yaxis.set_tick_params(which='both', direction='in')
    ax.tick_params(axis='x',labelsize=13,length=4,width=0.8,direction='in')
    Nh = np.array([10,15,20,30])
    IE = np.zeros(len(Nh),dtype=np.float64)
    for i in range(len(Nh)):
        RBM_Nh = np.loadtxt("t-step_Nh_"+str(Nh[i]), dtype=np.float64)
        RBM_Nh = RBM_Nh[RBM_Nh[:,0]<25]
        ax.plot(RBM_Nh[:,0]*unit_t,RBM_Nh[:,4],label=r'$N_{\rm h}='+str(Nh[i])+r'$',linewidth='1.',color=color[i])
        IE[i] = integral_error(RBM_Nh[:,0],RBM_Nh[:,j],ref[:,0],ref[:,j_q]) #?
    ax.set_ylim(2.e-10,5.0e-3)
    ax.set_xlim(0.5*unit_t,25*unit_t)
    ax.legend(bbox_to_anchor=(0.43,1.03),loc=1,prop={'size':13},handletextpad=0.5,handlelength=1.8,frameon=False)
    
    inset.set_xlabel(r'$N_{\rm h}$',fontdict={'family':fname2,'size': 14})
    # with latexmath():
    inset.set_ylabel(r'$ℰ_{I_{\rm R}}$',fontdict={'size': 15},fontweight='bold')
    # inset.set_xscale('log')
    inset.xaxis.set_tick_params(direction='in')
    inset.yaxis.set_tick_params(direction='in')
    inset.scatter(Nh,IE,marker='^',color=color[-1]) # ms=4.5
    x,y,k = power_law_fit(Nh,IE)
    k = round(k,2)
    inset.plot(x,y,linewidth='1',linestyle='--',color=color[-1],label='fit')
    inset.text(10,0.0135,r'$N_{\rm h}^{'+str(k)+r'}$',color=color[-1],fontsize=13)
    inset.legend(bbox_to_anchor=(1.0,1.05),loc=1,prop={'size':12},handletextpad=0.5,handlelength=1.5,frameon=False)
    
    # inset.set_ylim(-0.05,0.3)
    # inset.set_xlim(4*unit_t,15*unit_t)

plot_Nh(ax_c,inset_c)

# No,Nv,Nb = 2,1,2
# M,Nh,Na = 4,30,30
# Nd = 2 * No * Nv * Nb * M
# Ns = No * Nv
# nparameters = 2*Ns+2*Nh+1*Nd+Na+2*(1*2*Nd*Ns+Ns*Nh+Ns*Na+2*Nh*Nd)+1*Na*Nd+0*Ns*Ns
N_E = 8*np.array([4,5,6,7,8])
N_quick = np.array([6724,12884,21988,34612,51332])/10000
N_rbm = np.array([3712,4584,5456,7783,10510])/10000
ax_e.set_xlabel(r'$N_{\rm E}$',fontsize=16)
ax_e.text(-0.15,0.15,r'$N_{\rm para}$',transform=ax_e.transAxes,color=color[-1],rotation=90,fontsize=16)
ax_e.text(-0.15,0.35,r'${\rm /} \ N_{\rm RDT}$ ($\times 10^4$)',transform=ax_e.transAxes,color="black",rotation=90,fontsize=16)

# ax_e.text(0.8,0.22,transform=ax_e.transAxes,color="red")
# ax_e.set_ylabel(r'${N_{\rm para}}$$\  {\rm /} \ N_{\rm RDT}$ ($\times 10^4$)',fontdict={'family':fname2,'size': 14})
ax_e.scatter(N_E,N_rbm,marker='^',label=r'${\rm RBM}$',color=color[-1])
x,y,k = power_law_fit(N_E, N_rbm)
k = round(k,2)
ax_e.plot(x,y,'--',label='RBM fit',color=color[-1],linewidth='1.2')
ax_e.text(0.8,0.22,r'$N_{\rm E}^{'+str(k)+r'}$',fontdict={'family':fname2,'size': 16},transform=ax_e.transAxes,color=color[-1])

ax_e.scatter(N_E,N_quick,marker='o',label=r'${\rm ref.}$',color='black')

    # x,y,k = power_law_fit(Nh,IE,)
    # k = round(k,2)
    # inset.plot(x,y,linewidth='1',linestyle='--',color=color[-1],label='fit')
    # inset.text(9,0.02,r'$N_{\rm h}^{'+str(k)+r'}$',color=color[-1])

x,y,k = power_law_fit(N_E, N_quick)
k = round(k,2)
ax_e.plot(x,y,'--',color='black',label='ref. fit')
ax_e.text(0.75,0.85,r'$N_{\rm E}^{'+str(k)+r'}$',fontdict={'family':fname2,'size': 16},transform=ax_e.transAxes,color='black')

ax_e.tick_params(axis='y',labelsize=13,length=3,width=0.8,direction='in')
ax_e.tick_params(axis='x',labelsize=13,length=3,width=0.8,direction='in')
ax_e.legend(loc='upper left',fontsize=14,frameon=False)
# N_E = 8*np.array([4,5,6])
# N_quick = np.array([30466,58298,99402])/10000
# N_rbm = np.array([9652,11868,16674])/10000
# ax_e.set_xlabel(r'$N_{\rm E}$',fontsize=15)
# ax_e.set_ylabel(r'$N$ ($10^4$)',fontdict={'family':fname2,'size': 15})
# ax_e.plot(N_E,N_rbm,'^-',label=r'${\rm RBM}$',color=color[1],linewidth='1.2')
# ax_e.plot(N_E,N_quick,'o-',label=r'${\rm HEOM}$',color='black')


def plot_MC(ax:Axes,inset:Axes,j=7,t_max=25.):
    #j=7:I_r; j=1:n_up
    ax.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 18})
    ax.set_ylabel(r'$I_{\rm R}$ ($\Gamma$)',fontdict={'family':fname2,'size': 17})
    ax.set_ylim(-0.048,0.295)
    ax.set_xlim(4*unit_t,21.5*unit_t)
    ax.plot(quick['T=0.03'][:,0]*unit_t,quick['T=0.03'][:,4]*unit_I,linewidth='1.0',color=color[0],linestyle='--',zorder=0,label='ref.')
    ref = quick['T=0.03']
    # condition = ref[:,0]<t_max
    # ref = ref[condition]
    N_mc = [1024,1536,2048,4096,6144]
    mc_error = np.zeros(len(N_mc),dtype=np.float64)
    for i in range(len(N_mc)):
        MC = np.loadtxt('t-step_'+str(N_mc[i]))
        MC = MC[MC[:,0]<25]
        ax.plot(MC[:,0]*unit_t,MC[:,j]*unit_I,linewidth='1.2',color=color[i+1],label=r'$N_{\rm MC}=$'+str(N_mc[i]))
        if i==0:
            # scale0 = ref[:,0]>MC[0,0]
            # ref = ref[scale0]
            # scale1 = ref[:,0]<MC[-1,0]
            # ref = ref[scale1]
            ref_x = ref[:,0]
            ref_y = ref[:,4]
        rbm_x = MC[:,0]
        rbm_y = MC[:,j]
        # f = scipy.interpolate.interp1d(rbm_x,rbm_y)
        # y_rbm = f(ref_x)
        # y_quick = ref_y
        # delta = integral_error_(y_rbm,y_quick)
        delta = integral_error(rbm_x,rbm_y,ref_x,ref_y)
        mc_error[i] = delta
    # print(f'T={N_mc[i]}:{delta:.4e}')
    ax.legend(bbox_to_anchor=(0.55,1.0),loc=2,prop={'size':11.5},ncol=1,frameon=False)

    inset.set_xlabel(r'$N_{\rm MC}$',fontdict={'family':fname2,'size': 13})
    inset.set_ylabel(r'$ℰ_{I_{\rm R}}$',fontdict={'family':fname2,'size': 15})
    inset.xaxis.set_tick_params(direction='in')
    inset.yaxis.set_tick_params(which='both',direction='in')
    inset.set_yscale('log')
    inset.set_ylim(8e-3,2.)
    inset.plot(N_mc,mc_error,linewidth='1',marker='^',ms=4.5,color=color[-1],zorder=2) # ms=4.5
    inset.legend(bbox_to_anchor=(-0.03,1.05),loc=2,prop={'size':10},handletextpad=0.5,handlelength=1.5,frameon=False)
    np.savetxt("mc_error.txt",mc_error)

plot_MC(ax_d,inset_d)


# for i in range(len(listT_num)):
#     scale0 = rbm['T='+listT[i]][:,0]<= 25
#     # time_integral
#     rbm_x,rbm_y = rbm['T='+listT[i]][:,0][scale0],rbm['T='+listT[i]][:,7][scale0]
#     scale1 = quick['T='+listT[i]][:,0]<=rbm_x[-1]
#     ref_x,ref_y = quick['T='+listT[i]][:,0][scale1],quick['T='+listT[i]][:,4][scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm_y)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')


print('1')
ax_b.tick_params(axis='y',labelsize=14,length=3,width=0.8,direction='in')
ax_b.tick_params(axis='x',labelsize=14,length=3,width=0.8,direction='in')
ax_b.set_ylim(-0.05,2.0)
ax_b.set_xlim(0,25*unit_t)
inset_b.tick_params(axis='y',labelsize=8,length=2,width=0.8,direction='in')
inset_b.tick_params(axis='x',labelsize=7,length=2,width=0.8,direction='in')
inset_b.set_yticks([0.02,0.03,0.04,0.05])
inset_b.set_ylim(0.015,0.05)
# ax_b.tick_params(axis='y',labelsize=13,length=4,width=0.8,direction='in')
# ax_b.tick_params(axis='x',labelsize=13,length=4,width=0.8,direction='in')
# ax_b.set_xlim(0,25*unit_t)
ax_d.tick_params(axis='y',labelsize=13,length=3,width=0.8,direction='in')
ax_d.tick_params(axis='x',labelsize=13,length=3,width=0.8,direction='in')




plt.savefig('case1_all_kbT_1_1.png',dpi=100)
plt.savefig('case1_all_kbT_1_1.pdf',format="pdf")
# sys.exit()
# plt.savefig('case1_all_mc.eps',format='eps')
# sys.exit()
# plt.savefig('figure.svg', format='svg')#输入的图不支持矢量读取










# # T_j_cut4 = np.loadtxt("T_j_cut4", usecols=(2), dtype = np.float64)
# # T_inverse_cut4 = [(1/0.3+i*(1/0.03-1/0.3)/100) for i in range(len(T_j_cut4))]
# # T_j_cut5 = np.loadtxt("T_j_cut5", usecols=(2), dtype = np.float64)
# # T_inverse_cut5 = [(1/0.3+i*(1/0.03-1/0.3)/20) for i in range(len(T_j_cut5))]
# # T_j_cut6 = np.loadtxt("T_j_cut6", usecols=(2), dtype = np.float64)
# # T_inverse_cut6 = [(1/0.3+i*(1/0.03-1/0.3)/20) for i in range(len(T_j_cut6))]

# plt.figure('inset')
# # plt.plot(1/T,I_long,linewidth='0',marker='^',ms=6,color=color[-1],zorder=2) # ms=4.5
# # plt.plot(1/T,I_quick,linewidth='0',marker='o',ms=6,color=color[0],zorder=1) # ms=4.5
# # plt.plot(T_inverse_cut4,T_j_cut4,linewidth='1.6',label='cut4',color=color[0],linestyle='--',zorder=1)
# # plt.plot(T_inverse_cut5,T_j_cut5,linewidth='1.2',label='cut5',color=color[0],linestyle='--',zorder=1)
# # plt.plot(T_inverse_cut6,T_j_cut6,linewidth='0.8',label='cut6',color=color[0],linestyle='--',zorder=1)

# # plt.rcParams['mathtext.rm'] = 'Symbol'
# plt.rcParams['mathtext.rm'] = 'Arial'
# plt.rcParams['mathtext.it'] = 'Arial:italic'
# plt.axis('off')
# plt.text(0,0,s=r'$\langle \vec{n} \| \| \vec{n} \rangle$',fontsize=50,color=color[0],ha='left',va='baseline')

# # plt.plot([],[],label='RBM',linewidth='0',marker='^',ms=6,color=color[-1])
# # # inset_b.plot([],[],label='ref.',linestyle='dashed',linewidth='0.8',marker='o',ms=4.5,color=color[0])
# # plt.plot([],[],label='ref.',linewidth='0',marker='o',ms=6,color=color[0])
# # # inset_b.legend(bbox_to_anchor=(-0.03,1.05),loc=2,prop={'size':10},handletextpad=0.2,handlelength=1.5,frameon=False)
# # plt.legend(bbox_to_anchor=(-0.00,1.00),loc=2,prop={'size':16},frameon=False)
# plt.savefig('inset.png', dpi=200, bbox_inches='tight')




