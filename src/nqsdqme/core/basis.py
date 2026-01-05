import numpy as np

class Basis():
    """
    Class for the basis. 
    """
    def __init__(self,liouville,nonmccut=2,allcut=3,ifread=True):
        self.nonmccut = nonmccut
        self.allcut = allcut
        self.states = self.basis_generator(liouville,ifread)
        self.N_exact = self.states.shape[0]

        self.Nd = liouville.Nd
        self.Nbs =liouville.Nbs
        self.Ns = liouville.Ns
        self.env = liouville.env
        self.nstate = liouville.nstate
        self.nspin  = liouville.nspin
        self.nl = liouville.nvar
        self.sys_mode = liouville.sys_mode


    def basis_generator(self,liouville,ifread=True):
        if ifread:
            filename="table_cut"+str(self.nonmccut+1)+".data"
            statescols = [i for i in range(3,3+liouville.nstate)]
            states = np.loadtxt(filename,usecols=statescols)
        else:
            pass
        return states
