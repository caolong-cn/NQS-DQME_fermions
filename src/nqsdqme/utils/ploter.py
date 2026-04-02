import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import sys

#styles settings
class LineSettings():
    def __init__(self,data:np.array,linestyle='-',color='black',linewidth='1.',label=None,):
        self.data = data
        self.linestyle = linestyle
        self.color = color
        self.linewidth = linewidth
        self.label = label


class FigureSettings():
    def __init__(self,loc=0,xlabel=None,ylabel=None,title=None,xlim=(None,None),ylim=(None,None)):
        #legend,ylabel,xlabel,title,xlim,ylim
        self.loc = loc
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.xlim = xlim
        self.ylim = ylim

def plot(ax:Axes,lines,fset:FigureSettings):
    for line in lines:
        ax.plot(line.data[:,0],line.data[:,1],
                linestyle = line.linestyle, linewidth= line.linewidth,
                color = line.color,label = line.label)
    ax.legend(loc = fset.loc)
    ax.set_xlabel(fset.xlabel)
    ax.set_ylabel(fset.ylabel)
    ax.set_xlim(fset.xlim[0],fset.xlim[1])
    ax.set_ylim(fset.ylim[0],fset.ylim[1])
        #legend,ax1.set_ylabel(r'$n_{\rm \uparrow}$',fontdict={'family':fname2,'size': 12})
# ax1.text(-0.14,1.01,r'${\bf (a)}$',fontdict={'family':fname2,'size': 12},transform=ax1.transAxes)
# ax1.set_xlim(3.5*unit_t,25*unit_t)
# ax1.set_ylim(0.3,0.53)
# ax1.set_title(r'$k_{\rm B}T=3.0\,\Gamma$')
