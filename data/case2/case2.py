import numpy as np
import scipy
# import scipy.linalg 
import matplotlib.pyplot as plt
import sys

from scipy.signal import find_peaks

from derivative import non_uniform_five_point_derivative,integral_error

# import shutil
# import matplotlib
# from matplotlib import font_manager  
# shutil.rmtree(matplotlib.get_cachedir())
# for font in font_manager.fontManager.ttflist:  
#     print(font.name, '-', font.fname)



# # fname = 'Latin Modern Roman'
# fname = 'Cambria'
# # fname_italic = 'Latin Modern Roman:italic'
# fname_italic = 'Cambria:italic'
fname = 'Arial'
fname_italic = 'Arial:italic'
fname2 = 'Arial'

plt.rcParams['font.family']=fname  #使得坐标轴刻度标签字体变化
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = fname
plt.rcParams['mathtext.it'] = fname_italic

#unit
delta = 0.25e0 / 2      #meV
listT_unit = ['2.4','0.48','0.40','0.32','0.24']
meV = 1.6021766208e-19*1e-3   #J
h = 6.62607015e-34      #J·s
e = 1.602176634e-19     #C
pi = 3.1415926535897932
unit_t = 1e-12 * (delta * meV) / h *2*pi   #hbar/\gamma
unit_I = 1e-12 * h / (e * delta * meV) /(2*pi)   #e\gamma/hbar
coef = 0.658211928

t_long = 15.72
# print(15.72*unit_t)
# sys.exit()


#Temp
listT = ['0.3','0.06','0.05','0.04','0.03']
listT_num = ['03','006','005','004','003']
listT_num_Ehyb = ['30','6','5','4','3']
rbm_tstep, quick_ddot = {}, {}
rbm_Esys, quick_energy = {}, {}  #listT_rho0 ——> Esys_T
rbm_SvN, quick_SvN = {}, {}
rbm_Ehyb, quick_Ehyb = {}, {}
for i in range(len(listT_num)):
    rbm_tstep['T='+listT[i]] = np.loadtxt("rbm_t-step_T="+listT[i], usecols=(0,9), dtype=np.float64)
    quick_ddot['T='+listT[i]] = np.loadtxt("quick_ddot_T="+listT[i], usecols=(0,1), dtype=np.float64)
    rbm_Esys['T='+listT[i]] = np.loadtxt("rbm_Esys_T="+listT[i], dtype=np.float64)
    quick_energy['T='+listT[i]] = np.loadtxt("quick_engy_T="+listT[i], dtype=np.float64)
    rbm_SvN['T='+listT[i]] = np.loadtxt("rbm_SvN_T="+listT[i], dtype=np.float64)
    quick_SvN['T='+listT[i]] = np.loadtxt("quick_SvN_T="+listT[i], dtype=np.float64)
    rbm_Ehyb['T='+listT[i]] = np.loadtxt('E'+listT_num_Ehyb[i]+'.data', usecols=(0,2), dtype=np.float64)
    quick_Ehyb['T='+listT[i]] = np.loadtxt(listT_num_Ehyb[i]+'.data', usecols=(0,2), dtype=np.float64)

S12_long = [0.175725, 0.209782, 0.208038, 0.210039, 0.206815]
Esys_long = [-0.149124E+001, -0.159661E+001, -0.159110E+001, -0.159772E+001, -0.158743E+001]
SvN_long = [1.973374, 1.697501, 1.713769, 1.694377, 1.724540]
quick_Ehyb_long = np.array([-0.648349,-0.622367,-0.549918,-0.552599,-0.528591,-0.555389,-0.516658])
quick_Ehyb_0 = np.array([-0.709168,-0.707623,-0.660182,-0.675112,-0.656146,-0.687783,-0.657488])
delta_Ehyb_quick = quick_Ehyb_long - quick_Ehyb_0
rbm_Ehyb_long = coef*np.array([-9.846924e-01,-8.393293e-01,-8.030181e-01,-8.435039e-01,-7.839445e-01])
rbm_Ehyb_0 = coef*np.array([-1.077415e+00,-1.025594e+00,-9.968805e-01,-1.044952e+00,-9.988757e-01])
delta_Ehyb_rbm = rbm_Ehyb_long - rbm_Ehyb_0
#tlong = 25*unit = 4.7477107
T_rbm = np.array([2.4,0.48,0.4,0.32,0.24]) # demension Delta
T_quick = np.array([0.3,0.2,0.1,0.06,0.05,0.04,0.03]) # 
# color = ['black','#194f97','#625ba1','#00994e','#bd6b08','#c82d31']#black,blue,purple,green,yellow,red
# color = ['black','#9E7CA3','#5C2366','#428EA8','#2C5AB6','#c82d31']#黑, 淡紫, 深紫, 青, 蓝, 红, 
color = ['black','#2C5AB6','#428EA8','#6B2876','#9A759F','#c82d31']#黑, 蓝, 青, 深紫, 淡紫, 红






#general layout
fig = plt.figure(figsize=(8.6,7))
ax_a = fig.add_axes([0.08,0.56,0.88,0.38])
ax_a.axis('off')
ax_a.text(-0.04,0.95,r'${\bf (a)}$',fontdict={'family':fname2,'size': 15},transform=ax_a.transAxes)
ax_b = fig.add_axes([0.085,0.09,0.39,0.42])
ax_b.text(-0.08,1.05,r'${\bf (b)}$',fontdict={'family':fname2,'size': 15},transform=ax_b.transAxes)
ax_c = fig.add_axes([0.58,0.09,0.39,0.42])
ax_c.text(-0.08,1.05,r'${\bf (c)}$',fontdict={'family':fname2,'size': 15},transform=ax_c.transAxes)

inset_c = fig.add_axes([0.723,0.15,0.165,0.094])

#case2示意图a
# case2 = plt.imread('case2_a.png')
# ax_a.imshow(case2,cmap='hot')
# ax_a.set_xticks(())
# ax_a.set_yticks(())


#figure b(dS/dt)
ax_b_list = {}
legend_list = {}
ax_b.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 16})
# ax_b.set_ylabel(r'${\rm d} S_{\rm 12} {\rm /d}t$ ($\Gamma$)',fontsize=15)
ax_b.set_ylabel(r'$\dot{S}_{\rm 12}$ ($\Gamma$)',fontsize=15)
print('dS/dt:')
for i in range(len(listT_num)-0):
    t_diff, t_diff_x = non_uniform_five_point_derivative(rbm_tstep['T='+listT[i]][:,1], rbm_tstep['T='+listT[i]][:,0])
    ax_b_list[f'b{listT_num[i]}_1'], = ax_b.plot(t_diff_x*unit_t,(t_diff+0.04*i)/unit_t,linewidth='0.8',color=color[i+1],zorder=2)
    ddot_diff, ddot_diff_x = non_uniform_five_point_derivative(quick_ddot['T='+listT[i]][:,1], quick_ddot['T='+listT[i]][:,0])
    ax_b_list[f'b{listT_num[i]}_2'], = ax_b.plot(ddot_diff_x*unit_t,(ddot_diff+0.04*i)/unit_t,linewidth='0.8',color=color[i+1],linestyle='--',zorder=1)
    # print scipy.interpolate.interp1d
#     scale0 = (t_diff_x<=t_long) #t(1/Gamma) ~3.
#     rbm_x,rbm = t_diff_x[scale0],t_diff[scale0]
#     scale1 = ddot_diff_x<=rbm_x[-1]
#     ref_x,ref_y = ddot_diff_x[scale1],ddot_diff[scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
# print('dS/dt:')
# for i in range(len(listT_num)):
#     t_diff = np.diff(rbm_tstep['T='+listT[i]][:,1])/np.diff(rbm_tstep['T='+listT[i]][:,0])
#     t_diff_x = (rbm_tstep['T='+listT[i]][:,0][:-1]+rbm_tstep['T='+listT[i]][:,0][1:])/2
#     ax_b_list[f'b{listT_num[i]}_1'], = ax_b.plot(t_diff_x[0:]*unit_t,(t_diff[0:]+0.04*i)/unit_t,linewidth='0.8',color=color[i+1],zorder=2)
#     ddot_diff = np.diff(quick_ddot['T='+listT[i]][:,1])/np.diff(quick_ddot['T='+listT[i]][:,0])
#     ddot_diff_x = (quick_ddot['T='+listT[i]][:,0][:-1]+quick_ddot['T='+listT[i]][:,0][1:])/2
#     ax_b_list[f'b{listT_num[i]}_2'], = ax_b.plot(ddot_diff_x[0:]*unit_t,(ddot_diff[0:]+0.04*i)/unit_t,linewidth='0.8',color=color[i+1],linestyle='--',zorder=1)
#     # print scipy.interpolate.interp1d
#     scale0 = (t_diff_x<=t_long)
#     rbm_x,rbm = t_diff_x[scale0],t_diff[scale0]
#     scale1 = ddot_diff_x<=rbm_x[-1]
#     ref_x,ref_y = ddot_diff_x[scale1],ddot_diff[scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     # print(np.sum(np.abs(y_rbm-y_quick)*0.02))
#     # print(np.sum(np.abs(y_quick)*0.02))
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit() 
# print('S12:')
# for i in range(len(listT_num)):   #S12
#     # print scipy.interpolate.interp1d
#     scale0 = rbm_tstep['T='+listT[i]][:,0]<=t_long #t(1/Gamma)~3.
#     rbm_x,rbm = rbm_tstep['T='+listT[i]][:,0][scale0],rbm_tstep['T='+listT[i]][:,1][scale0]
#     scale1 = quick_ddot['T='+listT[i]][:,0]<=rbm_x[-1]
#     ref_x,ref_y = quick_ddot['T='+listT[i]][:,0][scale1],quick_ddot['T='+listT[i]][:,1][scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()


for i in range(len(listT_num)):
    ax_b.text(8*unit_t,(0.04*i+0.009)/unit_t,s=r'$k_{\rm B}T=$'+listT_unit[i]+r'$\Gamma$',fontsize=15,color=color[i+1],ha='left',va='baseline')
ax_b.plot([],[],label='RBM',linewidth='1',color=color[0])
ax_b.plot([],[],label='ref.',linewidth='1',color=color[0],linestyle='--')
# ax_b.plot([],[],label='RBM',linestyle='dashed',marker='o',ms=5.5,color=color[0])
# # ax_b.scatter([],[],label='RBM',color=color[0])
# ax_b.plot([],[],label='ref.',linewidth='0.8',color=color[0])#,linestyle='--'
# ax_b.legend(loc=1,prop={'size':15},ncol=1,frameon=False)
ax_b.legend(bbox_to_anchor=(1.0,1.02),loc=1,prop={'size':16},handletextpad=0.5,handlelength=1.8,frameon=False)



#figure c
# (Energy/SvN)
# ax_c_list = {}
# ax_c.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 16})
# # ax_c.set_ylabel(r'$E_{\rm sys}$',fontdict={'family':'Arial','size': 15})#(${\rm meV}$)
# # for i in range(len(listT_num)):
# #     ax_c_list['c{listT_num}_1'], = ax_c.plot(rbm_Esys['T='+listT[i]][:,0], rbm_Esys['T='+listT[i]][:,1]-Esys_long[i]+Esys_long[0]-0.12*i, linewidth='0.8', color=color[i+1], zorder=2)
# #     ax_c_list['c{listT_num}_2'], = ax_c.plot(quick_energy['T='+listT[i]][:,0], quick_energy['T='+listT[i]][:,1]-Esys_long[i]+Esys_long[0]-0.12*i, linewidth='0.8', color=color[i+1], linestyle='--', zorder=1)
# # for i in range(len(listT_num)):
# #     ax_c.text(8,Esys_long[0]-0.12*i+0.06,s=r'$T=$'+listT[i],fontsize=12,color=color[i+1],ha='left',va='baseline')
# ax_c.set_ylabel(r'$S_{\rm vN}$',fontdict={'family':fname2,'size': 16})
# for i in range(len(listT_num)):
#     ax_c_list['c{listT_num}_1'], = ax_c.plot(rbm_SvN['T='+listT[i]][:,0]*unit_t, rbm_SvN['T='+listT[i]][:,1]-SvN_long[i]+SvN_long[-1]+0.22*i, linewidth='0.8', color=color[i+1], zorder=2)
#     ax_c_list['c{listT_num}_2'], = ax_c.plot(quick_SvN['T='+listT[i]][:,0]*unit_t, quick_SvN['T='+listT[i]][:,1]-SvN_long[i]+SvN_long[-1]+0.22*i, linewidth='0.8', color=color[i+1], linestyle='--', zorder=1)
# for i in range(len(listT_num)):
#     ax_c.text(9*unit_t,SvN_long[-1]+0.223*i+0.04,s=r'$k_{\rm B}T=$'+listT_unit[i]+r'$\Gamma$',fontsize=15,color=color[i+1],ha='left',va='baseline')
# print('SvN:')
# for i in range(len(listT_num)):
#     # print scipy.interpolate.interp1d
#     scale0 = rbm_SvN['T='+listT[i]][:,0]<=t_long #t(1/Gamma)~3.
#     rbm_x,rbm = rbm_SvN['T='+listT[i]][:,0][scale0],rbm_SvN['T='+listT[i]][:,1][scale0]
#     scale1 = quick_SvN['T='+listT[i]][:,0]<=rbm_x[-1]
#     ref_x,ref_y = quick_SvN['T='+listT[i]][:,0][scale1],quick_SvN['T='+listT[i]][:,1][scale1]
#     scale2 = ref_x>=rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
# ax_c.plot([],[],label='RBM',linewidth='1',color=color[0])
# ax_c.plot([],[],label='ref.',linewidth='1',color=color[0],linestyle='--')
# # ax_c.plot([],[],label='RBM',linestyle='dashed',marker='o',ms=5,color=color[0])
# # # ax_c.scatter([],[],label='RBM',color=color[0])
# # ax_c.plot([],[],label='ref.',linewidth='0.8',color=color[0])
# # ax_c.legend(loc=1,prop={'size':16},ncol=1,frameon=False)
# ax_c.legend(bbox_to_anchor=(1.0,1.02),loc=1,prop={'size':16},handletextpad=0.5,handlelength=1.8,frameon=False)

# ax_c.set_ylabel(r'${\rm d} E_{\rm sys} {\rm /dt}$',fontdict={'family':'Arial','size': 15})
# for i in range(len(listT_num)):
#     t_diff = np.diff(rbm_Esys['T='+listT[i]][:,1])/np.diff(rbm_Esys['T='+listT[i]][:,0])
#     t_diff_x = (rbm_Esys['T='+listT[i]][:,0][:-1]+rbm_Esys['T='+listT[i]][:,0][1:])/2
#     ax_c.plot(t_diff_x[0:],t_diff[0:]+0.16-0.04*i,linewidth='0.8',color=color[i+1],zorder=2)
#     ddot_diff = np.diff(quick_energy['T='+listT[i]][:,1][::3])/np.diff(quick_energy['T='+listT[i]][:,0][::3])
#     ddot_diff_x = (quick_energy['T='+listT[i]][:,0][::3][:-1]+quick_energy['T='+listT[i]][:,0][::3][1:])/2
#     ax_c.plot(ddot_diff_x[0:],ddot_diff[0:]+0.16-0.04*i,linewidth='0.8',linestyle='--',color=color[i+1],zorder=1)
# for i in range(len(listT_num)):
#     ax_c.text(9,-0.02+0.16-0.04*i,s=r'$T=$'+listT[i],fontsize=12,color=color[i+1],ha='left',va='baseline')
# ax_c.plot([],[],label='RBM-HEOM',linewidth='0.8',color=color[0])
# ax_c.plot([],[],label='HEOM-QUICK',linewidth='0.8',color=color[0],linestyle='--')
# ax_c.legend(loc=4,prop={'size':12},ncol=1,frameon=False)


#plot Ehyb

ax_c.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 16})
ax_c.set_ylabel(r'$E_{\rm hyb}$ ($\Gamma$)',fontdict={'family':fname2,'size': 15})
Ehyb_long = np.array([-0.648349, -0.552599, -0.528591, -0.555389, -0.516658])/delta
for i in range(len(listT_num)):
    ax_c.plot(rbm_Ehyb['T='+listT[i]][:,0]*unit_t,rbm_Ehyb['T='+listT[i]][:,1]*coef/delta-Ehyb_long[i]-1.2+0.3*i,linewidth='1',color=color[i+1])
    ax_c.plot(quick_Ehyb['T='+listT[i]][:,0]*unit_t,quick_Ehyb['T='+listT[i]][:,1]/delta-Ehyb_long[i]-1.2+0.3*i,linewidth='1',linestyle='--',color=color[i+1])
    ax_c.text(14*unit_t,0.05-1.2+0.3*i,s=r'$k_{\rm B}T=$'+listT_unit[i]+r'$\Gamma$',fontsize=15,color=color[i+1],ha='left',va='baseline')
# ax_c.plot([],[],label='RBM',linewidth='1',color=color[0])
# ax_c.plot([],[],label='ref.',linewidth='1',color=color[0],linestyle='--')
# ax_c.legend(loc=4,fontsize=16,frameon=False)
# print('E_hyb:')
# #calculate integral error of Ehyb:
# for i in range(len(listT_num)):
#     scale0 = rbm_Ehyb['T='+listT[i]][:,0]<=t_long# t~3.
#     rbm_x,rbm = rbm_Ehyb['T='+listT[i]][:,0][scale0],rbm_Ehyb['T='+listT[i]][:,1][scale0]*coef
#     scale1 = quick_Ehyb['T='+listT[i]][:,0]<=rbm_x[-1]
#     ref_x,ref_y = quick_Ehyb['T='+listT[i]][:,0][scale1],quick_Ehyb['T='+listT[i]][:,1][scale1]
#     scale2 = ref_x>=0
#     # rbm_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_x,rbm)
#     y_rbm = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm,y_quick)
#     # print(np.sum(np.abs(y_rbm-y_quick)*0.02))
#     # print(np.sum(np.abs(y_quick-y_quick[-1])*0.02))
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
#inset_c
inset_c.set_xlabel(r'$\Gamma/k_{\rm B}T$',fontsize=11,labelpad=2)
# inset_c.set_xlabel(r'$k_{\rm B}T(\Gamma)$',fontsize=11,labelpad=2)
inset_c.set_ylabel(r'$\Delta E_{\rm hyb} (\Gamma)$',fontsize=11,labelpad=2)
# inset_c.set_ylabel(r'$I$',fontsize=12,labelpad=2)
# inset_c.plot(1/T,I_long,linewidth='1',color=color[0])
# inset_c.scatter(1/T,I_long,s=22,marker='^',color=color[-1],zorder=2)


T_Ehyb_cut4 = np.loadtxt("T-Ehyb_cut4_M6_final", usecols=(2), dtype = np.float64) -\
                np.loadtxt("T-Ehyb_cut4_M6_initial", usecols=(2), dtype = np.float64)
T_inverse_cut4 = np.array([(1/1.0+i*(1/0.025-1/1.0)/(len(T_Ehyb_cut4)-1)) for i in range(len(T_Ehyb_cut4))])
inset_c.plot(1/(T_rbm),delta_Ehyb_rbm/delta,linewidth='0.',marker='^',ms=4.5,color=color[-1],zorder=2) # ms=4.5
# inset_c.plot(1/(T_quick/delta),delta_Ehyb_quick/delta,linewidth='0.8',ms=4.5,color=color[0],zorder=1,linestyle = '-') # ms=4.5
print(delta*T_inverse_cut4)
inset_c.plot(delta*T_inverse_cut4[1:-3],T_Ehyb_cut4[1:-3]/delta,linewidth='1.0',linestyle='-',color=color[0],zorder=1)
# ,linestyle='--'
inset_c.plot([],[],label='RBM',linewidth='0',marker='^',ms=4.5,color=color[-1])
inset_c.plot([],[],label='ref.',linestyle='-',linewidth='1.',color=color[0])
# inset_c.plot([],[],label='ref.',linewidth='0',marker='o',ms=4.5,color=color[0])
inset_c.legend(bbox_to_anchor=(0.4,0.8),loc=2,prop={'size':10},handletextpad=0.5,handlelength=1.5,frameon=False)


ax_b.tick_params(axis='y',labelsize=12.5,length=2,width=0.8,direction='in')
ax_b.tick_params(axis='x',labelsize=12.5,length=2,width=0.8,direction='in')
ax_b.set_xlim(0,25*unit_t)
# ax_b.set_ylim(-0.01,0.245)
ax_c.tick_params(axis='y',labelsize=12.5,length=2,width=0.8,direction='in')
ax_c.tick_params(axis='x',labelsize=12.5,length=2,width=0.8,direction='in')
ax_c.set_xlim(0,25*unit_t)
ax_c.set_ylim(-2.4,0.24)

inset_c.tick_params(axis='y',labelsize=8,length=2,width=0.8,direction='in')
inset_c.tick_params(axis='x',labelsize=7,length=2,width=0.8,direction='in')
# inset_c.set_yticks([0.02,0.03,0.04,0.05])
inset_c.set_ylim(0.4,1.2)

plt.savefig('case2_all_kbT.png', dpi=400)
plt.savefig('case2_all_kbT.pdf')










listT = ['0.3']
listT_num = ['03']
tt = 25
# listT = ['0.03']
# listT_num = ['003']
# tt = 2

rbm_tstep_mc = {}
for i in range(len(listT_num)):
    rbm_tstep_mc['T='+listT[i]] = np.loadtxt("rbm_t-step_T="+listT[i]+'_mc', usecols=(0,9), dtype=np.float64)
plt.figure('e_S12_mc')
plt.xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':'Arial','size': 16})
plt.ylabel(r'$S_{\rm 12}$',fontdict={'family':'Arial','size': 15})
plt_list = {}
for i in range(len(listT_num)):
    plt_list[f'{listT_num[i]}_1'], = plt.plot(rbm_tstep['T='+listT[i]][:,0]*unit_t,rbm_tstep['T='+listT[i]][:,1],linewidth='0.8',color=color[i+1],zorder=2)
    plt_list[f'{listT_num[i]}_2'], = plt.plot(quick_ddot['T='+listT[i]][:,0]*unit_t,quick_ddot['T='+listT[i]][:,1],linewidth='0.8',color=color[i+1],linestyle='--',zorder=1)
    plt_list[f'{listT_num[i]}_3'], = plt.plot(rbm_tstep_mc['T='+listT[i]][:,0]*unit_t,rbm_tstep_mc['T='+listT[i]][:,1],linewidth='1.2',color=color[i+2],zorder=3)
# for i in range(len(listT_num)):
#     plt.text(10,0.033*i+0.185,s=r'$T=$'+listT[i],fontsize=15,color=color[i+1],ha='left',va='baseline')
# plt.plot([],[],label='RBM',linewidth='1',color=color[0])
# plt.plot([],[],label='ref.',linewidth='1',color=color[0],linestyle='--')
# plt.legend(loc=4,prop={'size':15},ncol=1,frameon=False)
plt.xlim(0,tt*unit_t)
plt.savefig('e_S12_mc.png', dpi=200, bbox_inches='tight')


# fig = plt.figure('mc_case2')
fig = plt.figure(figsize=(10,3.5))
ax_a = fig.add_axes([0.00,0.0,0.4,0.95])
ax_a.text(-0.08,1.05,r'${\bf (a)}$',fontdict={'family':fname2,'size': 16},transform=ax_a.transAxes)
ax_b = fig.add_axes([0.5,0.0,0.4,0.95])
ax_b.text(-0.08,1.05,r'${\bf (b)}$',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)

ax_a.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':'Arial','size': 18})
# ax_a.set_ylabel(r'${\rm d} S_{\rm 12} {\rm /d}t$ ($\Gamma$)',fontsize=17)
# ax_a.set_ylabel(r'\dot{S}_{\rm 12}$ ($\Gamma$)',fontsize=17)
plt_list = {}
for i in range(len(listT_num)):
    t_diff_mc = np.diff(rbm_tstep_mc['T='+listT[i]][:,1])/np.diff(rbm_tstep_mc['T='+listT[i]][:,0])
    t_diff_x_mc = (rbm_tstep_mc['T='+listT[i]][:,0][:-1]+rbm_tstep_mc['T='+listT[i]][:,0][1:])/2
    ax_a.plot(t_diff_x_mc[0:]*unit_t,t_diff_mc[0:]/unit_t,linewidth='0.35',color=color[5],zorder=1)
    t_diff = np.diff(rbm_tstep['T='+listT[i]][:,1])/np.diff(rbm_tstep['T='+listT[i]][:,0])
    t_diff_x = (rbm_tstep['T='+listT[i]][:,0][:-1]+rbm_tstep['T='+listT[i]][:,0][1:])/2
    ax_a.plot(t_diff_x[0:]*unit_t,t_diff[0:]/unit_t,linewidth='1.3',color=color[0],linestyle='--',zorder=2,dashes=(6,3))
    # ddot_diff = np.diff(quick_ddot['T='+listT[i]][:,1])/np.diff(quick_ddot['T='+listT[i]][:,0])
    # ddot_diff_x = (quick_ddot['T='+listT[i]][:,0][:-1]+quick_ddot['T='+listT[i]][:,0][1:])/2
    # ax_a.plot(ddot_diff_x[0:]*unit_t,ddot_diff[0:]/unit_t,linewidth='1',color=color[4*i+1],linestyle='--',zorder=2)

#     print('dS/dt_mc:')
#     scale0 = (t_diff_x_mc<=t_long) #t~3.
#     rbm_mc_x,rbm_mc = t_diff_x_mc[scale0],t_diff_mc[scale0]
#     scale1 = (t_diff_x<=rbm_mc_x[-1])
#     ref_x,ref_y = t_diff_x[scale1],t_diff[scale1]
#     # scale1 = ddot_diff_x<=rbm_mc_x[-1]
#     # ref_x,ref_y = ddot_diff_x[scale1],ddot_diff[scale1]
#     scale2 = ref_x>=rbm_mc_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_mc_x,rbm_mc)
#     y_rbm_mc = f(ref_x)
#     y_quick = ref_y
#     delta = integral_error(y_rbm_mc,y_quick)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
# print('S12_mc:')
# for i in range(len(listT_num)):   #S12
#     scale0 = rbm_tstep_mc['T='+listT[i]][:,0]<=t_long #t~3.
#     rbm_mc_x,rbm_mc = rbm_tstep_mc['T='+listT[i]][:,0][scale0],rbm_tstep_mc['T='+listT[i]][:,1][scale0]
#     scale1 = rbm_tstep['T='+listT[i]][:,0]<=rbm_mc_x[-1]
#     ref_x,ref_y = rbm_tstep['T='+listT[i]][:,0][scale1],rbm_tstep['T='+listT[i]][:,1][scale1]
#     # scale1 = quick_ddot['T='+listT[i]][:,0]<=rbm_mc_x[-1]
#     # ref_x,ref_y = quick_ddot['T='+listT[i]][:,0][scale1],quick_ddot['T='+listT[i]][:,1][scale1]
#     scale2 = ref_x>=rbm_mc_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_mc_x,rbm_mc)
#     y_rbm_mc = f(ref_x)
#     y_rbm = ref_y
#     delta = integral_error(y_rbm_mc,y_rbm)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
# for i in range(len(listT_num)):
#     plt.text(10,0.033*i+0.185,s=r'$T=$'+listT[i],fontsize=15,color=color[4*ii+1],ha='left',va='baseline')
ax_a.plot([],[],label='RBM (MCMC)',linewidth='1',color=color[5])
ax_a.plot([],[],label='RBM (full)',linewidth='1',color=color[0],linestyle='--',dashes=(5,2))
ax_a.legend(loc=1,fontsize=18,frameon=False)
ax_a.tick_params(axis='y',labelsize=13,length=3,width=1.2,direction='in')
ax_a.tick_params(axis='x',labelsize=13,length=3,width=1.2,direction='in')
ax_a.set_xlim(0,tt*unit_t)
ax_a.set_ylim(-0.05,0.4)
ax_a.set_yticks([0,0.1,0.2,0.3,0.4])

ax_b.set_xlabel(r'$t$ ($1/\Gamma$)',fontdict={'family':fname2,'size': 18})
ax_b.set_ylabel(r'$S_{\rm vN}$',fontdict={'family':fname2,'size': 18})
rbm_SvN_mc = {}
for i in range(len(listT_num)):
    rbm_SvN_mc['T='+listT[i]] = np.loadtxt("rbm_SvN_T="+listT[i]+'_mc', dtype=np.float64)
    ax_b.plot(rbm_SvN_mc['T='+listT[i]][:,0]*unit_t, rbm_SvN_mc['T='+listT[i]][:,1], linewidth='1.2', color=color[5], zorder=2)
    ax_b.plot(rbm_SvN['T='+listT[i]][:,0]*unit_t, rbm_SvN['T='+listT[i]][:,1], linewidth='1.2', color=color[0], linestyle='--', zorder=1,dashes=(5,2))
    # ax_b.plot(quick_SvN['T='+listT[i]][:,0]*unit_t, quick_SvN['T='+listT[i]][:,1], linewidth='0.8', color=color[i+1], linestyle='--', zorder=1)
# print('SvN_mc:')
# for i in range(len(listT_num)):
#     # print scipy.interpolate.interp1d
#     scale0 = rbm_SvN_mc['T='+listT[i]][:,0]<=t_long #t~3.
#     rbm_mc_x,rbm_mc = rbm_SvN_mc['T='+listT[i]][:,0][scale0],rbm_SvN_mc['T='+listT[i]][:,1][scale0]
#     scale1 = rbm_SvN['T='+listT[i]][:,0]<=rbm_mc_x[-1]
#     ref_x,ref_y = rbm_SvN['T='+listT[i]][:,0][scale1],rbm_SvN['T='+listT[i]][:,1][scale1]
#     # scale1 = quick_SvN['T='+listT[i]][:,0]<=rbm_mc_x[-1]
#     # ref_x,ref_y = quick_SvN['T='+listT[i]][:,0][scale1],quick_SvN['T='+listT[i]][:,1][scale1]
#     scale2 = ref_x>=rbm_mc_x[0]
#     ref_x,ref_y = ref_x[scale2],ref_y[scale2]
#     f = scipy.interpolate.interp1d(rbm_mc_x,rbm_mc)
#     y_rbm_mc = f(ref_x)
#     y_rbm = ref_y
#     delta = integral_error(y_rbm_mc,y_rbm)
#     print(f'T={listT[i]}:{delta:.4e}')
# sys.exit()
ax_b.plot([],[],label='RBM (MCMC)',linewidth='1',color=color[5])
ax_b.plot([],[],label='RBM (full)',linewidth='1',color=color[0],linestyle='--',dashes=(5,2))
ax_b.legend(loc=1,fontsize=18,frameon=False)
ax_b.tick_params(axis='y',labelsize=13,length=3,width=0.8,direction='in')
ax_b.tick_params(axis='x',labelsize=13,length=3,width=0.8,direction='in')
ax_b.set_xlim(0,25*unit_t)
ax_b.set_yticks([2.0,2.1,2.2])

plt.savefig('mc_case2_kbT.png', dpi=200, bbox_inches='tight')
plt.savefig('mc_case2_kbT.pdf',bbox_inches='tight')










# No,Nv,Nb = 2,2,1
# M,Nh,Na = 8,70,30
# Nd = 2 * No * Nv * Nb * M
# Ns = No * Nv
# nparameters = 2*Ns+2*Nh+1*Nd+Na+2*(1*2*Nd*Ns+Ns*Nh+Ns*Na+2*Nh*Nd)+1*Na*Nd+0*Ns*Ns
# N_E = 8*np.array([4,5,6,7,8])
# N_quick = np.array([30467,58298,99402,156580,231794])/10000
# N_rbm = np.array([9652,11868,16674,19290,25296])/10000
N_E = 8*np.array([4,5,6])
N_quick = np.array([30467,58298,99402])/10000
N_rbm = np.array([9652,11868,16674])/10000

plt.figure('N_para')
plt.xlabel(r'$N_{\rm E}$',fontsize=16)
plt.ylabel(r'$N_{\rm para}\ {\rm or} \ N_{\rm ADO}$ ($\times 10^4$)',fontdict={'family':fname2,'size': 15})
plt.plot(N_E,N_rbm,'^-',label=r'${\rm RBM}$',color=color[-1],linewidth='1.2')
plt.plot(N_E,N_quick,'o-',label=r'${\rm ref.}$',color='black')
plt.tick_params(axis='y',labelsize=13,length=3,width=0.8,direction='in')
plt.tick_params(axis='x',labelsize=13,length=3,width=0.8,direction='in')
plt.legend(loc=2,fontsize=18,frameon=False)
# plt.text(33,18,s='(case2)',fontsize=10,ha='left',va='baseline')
# plt.xlim(30,65)
plt.xticks([35,40,45])
# plt.ylim(0,25)
# plt.yticks([5,10,15,20])
# plt.savefig('N_para.png',dpi=200,bbox_inches='tight')
plt.savefig('N_para.pdf',bbox_inches='tight')



























# plt.figure('diff_Esys')
# plt.xlabel(r'$t$',fontdict={'family':'Arial','size': 16})
# plt.ylabel(r'${\rm d} E_{\rm sys} {\rm /dt}$',fontdict={'family':'Arial','size': 15})
# for i in range(len(listT_num)):
#     t_diff = np.diff(rbm_Esys['T='+listT[i]][:,1])/np.diff(rbm_Esys['T='+listT[i]][:,0])
#     t_diff_x = (rbm_Esys['T='+listT[i]][:,0][:-1]+rbm_Esys['T='+listT[i]][:,0][1:])/2
#     plt.plot(t_diff_x[0:],t_diff[0:]+0.2-0.05*i,linewidth='0.8',label=r'$T=$'+listT[i],color=color[i+1],zorder=2)
#     ddot_diff = np.diff(quick_energy['T='+listT[i]][:,1][::3])/np.diff(quick_energy['T='+listT[i]][:,0][::3])
#     ddot_diff_x = (quick_energy['T='+listT[i]][:,0][::3][:-1]+quick_energy['T='+listT[i]][:,0][::3][1:])/2
#     plt.plot(ddot_diff_x[0:],ddot_diff[0:]+0.2-0.05*i,linewidth='0.8',linestyle='--',color=color[i+1],zorder=1)
# plt.legend(loc='lower right',fontsize=12,frameon=False)
# plt.xlim(0,25)
# plt.savefig('diff_Esys.png', dpi=100, bbox_inches='tight')

# plt.figure('diff_SvN')
# plt.xlabel(r'$t$',fontdict={'family':'Arial','size': 16})
# plt.ylabel(r'${\rm d} S_{\rm vN} {\rm /dt}$',fontdict={'family':'Arial','size': 15})
# for i in range(len(listT_num)):
#     t_diff = np.diff(rbm_SvN['T='+listT[i]][:,1])/np.diff(rbm_SvN['T='+listT[i]][:,0])
#     t_diff_x = (rbm_SvN['T='+listT[i]][:,0][:-1]+rbm_SvN['T='+listT[i]][:,0][1:])/2
#     plt.plot(t_diff_x[0:],t_diff[0:]+0.2-0.05*i,linewidth='0.8',label=r'$T=$'+listT[i],color=color[i+1],zorder=2)
#     ddot_diff = np.diff(quick_SvN['T='+listT[i]][:,1])/np.diff(quick_SvN['T='+listT[i]][:,0])
#     ddot_diff_x = (quick_SvN['T='+listT[i]][:,0][:-1]+quick_SvN['T='+listT[i]][:,0][1:])/2
#     plt.plot(ddot_diff_x[0:],ddot_diff[0:]+0.2-0.05*i,linewidth='0.8',linestyle='--',color=color[i+1],zorder=1)
# plt.legend(loc='lower right',fontsize=12,frameon=False)
# plt.xlim(0,25)
# plt.savefig('diff_SvN.png', dpi=100, bbox_inches='tight')













# TT = np.array([0.3, 0.2, 0.1, 0.06, 0.05, 0.04, 0.03])
# S12_long = np.array([0.175725, 0.199744, 0.208666, 0.209782, 0.208038, 0.210039, 0.206815])
# Esys_long = np.array([-0.149124E+001, -0.156509E+001, -0.159220E+001, -0.159661E+001, -0.159110E+001, -0.159772E+001, -0.158743E+001])
# # M6
# # Esys_long = np.array([-0.149142E+001,-0.156584E+001,-0.159996E+001,-0.160196E+001,-0.160094E+001,-0.159775E+001,-0.158747E+001])
# SvN_long = np.array([1.973374, 1.787150, 1.710003, 1.697501, 1.713769, 1.694377, 1.724540])
# color = ['black','#194f97','#625ba1','#00994e','#bd6b08','#c82d31']#black,blue,purple,green,yellow,red
# plt.figure('dd T-steady',figsize=(7,5))
# # plt.xlabel(r'$T$',fontdict={'family':'Arial','size': 13})
# # plt.ylabel(r'${\rm operater}$',fontdict={'family':'Arial','size': 15})
# ttsxt = 'Esys'
# plt.ylabel(ttsxt,fontdict={'family':'Cambria','size': 16})
# plt.subplots_adjust(top=0.99,bottom=0.12,left=0.11,right=0.99,hspace=0.,wspace=0.)
# plt.xlabel('T',fontdict={'family':'Cambria','size': 15})
# plt.scatter(TT,Esys_long,label=ttsxt,color=color[-1])
# # plt.xlabel('1/T',fontdict={'family':'Cambria','size': 15})
# # plt.scatter(1/TT,Esys_long,label=ttsxt,color=color[-1])
# # plt.legend(loc='upper right',color='white',fontsize=12,frameon=False)
# # plt.xlim(0,25)
# plt.savefig('dd T-steady.png', dpi=200, bbox_inches='tight')






