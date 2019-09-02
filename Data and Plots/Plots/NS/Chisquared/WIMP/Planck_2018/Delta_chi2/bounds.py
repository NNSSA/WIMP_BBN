import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline

dataNUBE1 = np.loadtxt("../../Case=NU_Stat=BE_gDM=1._Sigmav=1._BR=0", skiprows=1)
dataNUBE2 = np.loadtxt("../../Case=NU_Stat=BE_gDM=2._Sigmav=1._BR=0", skiprows=1)
dataNUBE3 = np.loadtxt("../../Case=NU_Stat=BE_gDM=3._Sigmav=1._BR=0", skiprows=1)
dataNUFD2 = np.loadtxt("../../Case=NU_Stat=FD_gDM=2._Sigmav=1._BR=0", skiprows=1)
dataNUFD4 = np.loadtxt("../../Case=NU_Stat=FD_gDM=4._Sigmav=1._BR=0", skiprows=1)

massNUBE1 = np.unique(dataNUBE1[:,0])
Omegab_fullNUBE1 = dataNUBE1[:,1]
OmegabNUBE1 = np.unique(Omegab_fullNUBE1)
NeffNUBE1 = dataNUBE1[:,2]
HeNUBE1 = dataNUBE1[:,8]*4
DoverHNUBE1 = dataNUBE1[:,5]/dataNUBE1[:,4]

massNUBE2 = np.unique(dataNUBE2[:,0])
Omegab_fullNUBE2 = dataNUBE2[:,1]
OmegabNUBE2 = np.unique(Omegab_fullNUBE2)
NeffNUBE2 = dataNUBE2[:,2]
HeNUBE2 = dataNUBE2[:,8]*4
DoverHNUBE2 = dataNUBE2[:,5]/dataNUBE2[:,4]

massNUBE3 = np.unique(dataNUBE3[:,0])
Omegab_fullNUBE3 = dataNUBE3[:,1]
OmegabNUBE3 = np.unique(Omegab_fullNUBE3)
NeffNUBE3 = dataNUBE3[:,2]
HeNUBE3 = dataNUBE3[:,8]*4
DoverHNUBE3 = dataNUBE3[:,5]/dataNUBE3[:,4]

massNUFD2 = np.unique(dataNUFD2[:,0])
Omegab_fullNUFD2 = dataNUFD2[:,1]
OmegabNUFD2 = np.unique(Omegab_fullNUFD2)
NeffNUFD2 = dataNUFD2[:,2]
HeNUFD2 = dataNUFD2[:,8]*4
DoverHNUFD2 = dataNUFD2[:,5]/dataNUFD2[:,4]

massNUFD4 = np.unique(dataNUFD4[:,0])
Omegab_fullNUFD4 = dataNUFD4[:,1]
OmegabNUFD4 = np.unique(Omegab_fullNUFD4)
NeffNUFD4 = dataNUFD4[:,2]
HeNUFD4 = dataNUFD4[:,8]*4
DoverHNUFD4 = dataNUFD4[:,5]/dataNUFD4[:,4]

names = ["Neutrinophilic - Boson - g = 1", "Neutrinophilic - Boson - g = 2", "Neutrinophilic - Boson - g = 3", "Neutrinophilic - Fermion - g = 2", "Neutrinophilic - Fermion - g = 4"]
masses = [massNUBE1, massNUBE2, massNUBE3, massNUFD2, massNUFD4]
Omegab_fulls = [Omegab_fullNUBE1, Omegab_fullNUBE2, Omegab_fullNUBE3, Omegab_fullNUFD2, Omegab_fullNUFD4]
Omegabs = [OmegabNUBE1, OmegabNUBE2, OmegabNUBE3, OmegabNUFD2, OmegabNUFD4]
Neffs = [NeffNUBE1, NeffNUBE2, NeffNUBE3, NeffNUFD2, NeffNUFD4]
Hes = [HeNUBE1, HeNUBE2, HeNUBE3, HeNUFD2, HeNUFD4]
DoverHs = [DoverHNUBE1, DoverHNUBE2, DoverHNUBE3, DoverHNUFD2, DoverHNUFD4]

stddev = np.array([0.954499736103642, 0.997300203936740])
deltaChisq = []
for j in range(len(stddev)):
    deltaChisq.append(chi2.ppf(stddev[j], 1))

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
    print("\n{}".format(names[j]))

    typ = 3
    for sig, cl in enumerate(deltaChisq):
        BBNInterp = UnivariateSpline(mass, DeltaChisqBBN-cl, s=0)
        CMBInterp = UnivariateSpline(mass, DeltaChisqCMB-cl, s=0)
        BBNCMBInterp = UnivariateSpline(mass, DeltaChisqBBNCMB-cl, s=0)
        print("\t{}sigma:\t BBN = {}\t CMB = {}\t BBN+CMB = {}\n".format(sig+2,BBNInterp.roots(), CMBInterp.roots(), BBNCMBInterp.roots()))

