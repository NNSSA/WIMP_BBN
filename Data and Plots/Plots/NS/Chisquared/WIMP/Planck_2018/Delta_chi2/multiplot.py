import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline

dataEEBE1 = np.loadtxt("../../Case=EE_Stat=BE_gDM=1._Sigmav=0._BR=0", skiprows=1)
dataEEBE2 = np.loadtxt("../../Case=EE_Stat=BE_gDM=2._Sigmav=0._BR=0", skiprows=1)
dataEEBE3 = np.loadtxt("../../Case=EE_Stat=BE_gDM=3._Sigmav=0._BR=0", skiprows=1)
dataEEFD2 = np.loadtxt("../../Case=EE_Stat=FD_gDM=2._Sigmav=0._BR=0", skiprows=1)
dataEEFD4 = np.loadtxt("../../Case=EE_Stat=FD_gDM=4._Sigmav=0._BR=0", skiprows=1)

massEEBE1 = np.unique(dataEEBE1[:,0])
Omegab_fullEEBE1 = dataEEBE1[:,1]
OmegabEEBE1 = np.unique(Omegab_fullEEBE1)
NeffEEBE1 = dataEEBE1[:,2]
HeEEBE1 = dataEEBE1[:,8]*4
DoverHEEBE1 = dataEEBE1[:,5]/dataEEBE1[:,4]

massEEBE2 = np.unique(dataEEBE2[:,0])
Omegab_fullEEBE2 = dataEEBE2[:,1]
OmegabEEBE2 = np.unique(Omegab_fullEEBE2)
NeffEEBE2 = dataEEBE2[:,2]
HeEEBE2 = dataEEBE2[:,8]*4
DoverHEEBE2 = dataEEBE2[:,5]/dataEEBE2[:,4]

massEEBE3 = np.unique(dataEEBE3[:,0])
Omegab_fullEEBE3 = dataEEBE3[:,1]
OmegabEEBE3 = np.unique(Omegab_fullEEBE3)
NeffEEBE3 = dataEEBE3[:,2]
HeEEBE3 = dataEEBE3[:,8]*4
DoverHEEBE3 = dataEEBE3[:,5]/dataEEBE3[:,4]

massEEFD2 = np.unique(dataEEFD2[:,0])
Omegab_fullEEFD2 = dataEEFD2[:,1]
OmegabEEFD2 = np.unique(Omegab_fullEEFD2)
NeffEEFD2 = dataEEFD2[:,2]
HeEEFD2 = dataEEFD2[:,8]*4
DoverHEEFD2 = dataEEFD2[:,5]/dataEEFD2[:,4]

massEEFD4 = np.unique(dataEEFD4[:,0])
Omegab_fullEEFD4 = dataEEFD4[:,1]
OmegabEEFD4 = np.unique(Omegab_fullEEFD4)
NeffEEFD4 = dataEEFD4[:,2]
HeEEFD4 = dataEEFD4[:,8]*4
DoverHEEFD4 = dataEEFD4[:,5]/dataEEFD4[:,4]

names = ["Electrophilic - Boson - g = 1", "Electrophilic - Boson - g = 2", "Electrophilic - Boson - g = 3", "Electrophilic - Fermion - g = 4"]
masses = [massEEBE1, massEEBE2, massEEBE3, massEEFD4]
Omegab_fulls = [Omegab_fullEEBE1, Omegab_fullEEBE2, Omegab_fullEEBE3, Omegab_fullEEFD4]
Omegabs = [OmegabEEBE1, OmegabEEBE2, OmegabEEBE3, OmegabEEFD4]
Neffs = [NeffEEBE1, NeffEEBE2, NeffEEBE3, NeffEEFD4]
Hes = [HeEEBE1, HeEEBE2, HeEEBE3, HeEEFD4]
DoverHs = [DoverHEEBE1, DoverHEEBE2, DoverHEEBE3, DoverHEEFD4]

stddev = np.array([0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856])
deltaChisq = []
for j in range(len(stddev)):
    deltaChisq.append(chi2.ppf(stddev[j], 1))


plt.figure(figsize=(20,20))
for j in range(len(names)):

    He = Hes[j]
    DoverH = DoverHs[j]
    Omegab = Omegabs[j]
    mass = masses[j]
    Omegab_full = Omegab_fulls[j]
    Neff = Neffs[j]

    # BBN
    ###########################################################################################################

    sigmasqHe = (0.003)**2 + (0.00017)**2
    sigmasqDoverH = (0.027e-5)**2 + (0.036e-5)**2

    MeasuredHe = 0.245
    MeasuredDoverH = 2.569e-5

    ChisqBBN = (He - MeasuredHe)**2 / sigmasqHe + (DoverH - MeasuredDoverH)**2 / sigmasqDoverH
    MarginalizedChisqBBN = [np.min(ChisqBBN[i * len(Omegab):(i+1) * len(Omegab)]) for i in range(len(mass))]
    MinimumChisqBBN = np.min(ChisqBBN)
    DeltaChisqBBN = MarginalizedChisqBBN - MinimumChisqBBN

    ###########################################################################################################


    # CMB
    ###########################################################################################################

    ChisqCMB = (Omegab_full - 0.02225) * ((Omegab_full - 0.02225)*4.66287e7 + (Neff - 2.89)*-33110.3 + (He - 0.246)*-496043.) + (Neff - 2.89) * ((Omegab_full - 0.02225)*-33110.3 + (Neff - 2.89)*43.3733 + (He - 0.246)*588.262) + (He - 0.246)*((Omegab_full - 0.02225)*-496043. + (Neff - 2.89)*588.262 + (He - 0.246)*11168.2)
    MarginalizedChisqCMB = [np.min(ChisqCMB[i * len(Omegab):(i+1) * len(Omegab)]) for i in range(len(mass))]
    MinimumChisqCMB = np.min(ChisqCMB)
    DeltaChisqCMB = MarginalizedChisqCMB - MinimumChisqCMB

    ###########################################################################################################


    # BBN + CMB
    ###########################################################################################################

    ChisqBBNCMB = ChisqBBN + ChisqCMB
    MarginalizedChisqBBNCMB = [np.min(ChisqBBNCMB[i * len(Omegab):(i+1) * len(Omegab)]) for i in range(len(mass))]
    MinimumChisqBBNCMB = np.min(ChisqBBNCMB)
    DeltaChisqBBNCMB = MarginalizedChisqBBNCMB - MinimumChisqBBNCMB

    ###########################################################################################################

    plt.subplot(2,2,j+1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    plt.plot(mass, DeltaChisqBBN, linewidth=3, linestyle="dashdot", color="black", label="BBN")
    plt.plot(mass, DeltaChisqCMB, linewidth=3, linestyle="dotted", color="black", label="CMB")
    plt.plot(mass, DeltaChisqBBNCMB, linewidth=3, linestyle="solid", color="black", label="BBN+CMB")
    for i in range(len(deltaChisq)):
        plt.axhline(deltaChisq[i])
        plt.annotate(r"{a:d}$\sigma$".format(a=i+1), (0.11, deltaChisq[i]+0.2), color="black", fontsize=20)

    plt.axis(ymin=0, ymax=40, xmin=1e-1, xmax=14)
    plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
    plt.ylabel(r"$\Delta\tilde{\chi}^2$", fontsize=20)
    plt.semilogx()
    plt.title("{}".format(names[j]), fontsize=20)
    plt.legend(loc='upper right', prop={'size': 16}, frameon=False, markerfirst=False)


    ###########################################################################################################

plt.savefig("Electrophilic_multiplot")

plt.show()
# typ = 3
# for sig, cl in enumerate(deltaChisq):
#     BBNInterp = UnivariateSpline(mass, DeltaChisqBBN-cl, s=0)
#     CMBInterp = UnivariateSpline(mass, DeltaChisqCMB-cl, s=0)
#     BBNCMBInterp = UnivariateSpline(mass, DeltaChisqBBNCMB-cl, s=0)
#     print("{}sigma:\t BBN = {}\t CMB = {}\t BBN+CMB = {}\n".format(sig+1,BBNInterp.roots(), CMBInterp.roots(), BBNCMBInterp.roots()))
