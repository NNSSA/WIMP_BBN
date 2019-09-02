import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline

dataEEFD2 = np.loadtxt("../../Case=EE_Stat=FD_gDM=2._Sigmav=0._BR=0", skiprows=1)
massEEFD2 = np.unique(dataEEFD2[:,0])
Omegab_fullEEFD2 = dataEEFD2[:,1]
OmegabEEFD2 = np.unique(Omegab_fullEEFD2)
NeffEEFD2 = dataEEFD2[:,2]
HeEEFD2 = dataEEFD2[:,8]*4
DoverHEEFD2 = dataEEFD2[:,5]/dataEEFD2[:,4]

dataNUFD2 = np.loadtxt("../../Case=NU_Stat=FD_gDM=2._Sigmav=1._BR=0", skiprows=1)
massNUFD2 = np.unique(dataNUFD2[:,0])
Omegab_fullNUFD2 = dataNUFD2[:,1]
OmegabNUFD2 = np.unique(Omegab_fullNUFD2)
NeffNUFD2 = dataNUFD2[:,2]
HeNUFD2 = dataNUFD2[:,8]*4
DoverHNUFD2 = dataNUFD2[:,5]/dataNUFD2[:,4]


# BBN
###########################################################################################################

sigmasqHe = (0.003)**2 + (0.00017)**2
sigmasqDoverH = (0.027e-5)**2 + (0.036e-5)**2

MeasuredHe = 0.245
MeasuredDoverH = 2.569e-5

ChisqBBNEEFD2 = (HeEEFD2 - MeasuredHe)**2 / sigmasqHe + (DoverHEEFD2 - MeasuredDoverH)**2 / sigmasqDoverH
MarginalizedChisqBBNEEFD2 = [np.min(ChisqBBNEEFD2[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqBBNEEFD2 = np.min(ChisqBBNEEFD2)
DeltaChisqBBNEEFD2 = MarginalizedChisqBBNEEFD2 - MinimumChisqBBNEEFD2

ChisqBBNNUFD2 = (HeNUFD2 - MeasuredHe)**2 / sigmasqHe + (DoverHNUFD2 - MeasuredDoverH)**2 / sigmasqDoverH
MarginalizedChisqBBNNUFD2 = [np.min(ChisqBBNNUFD2[i * len(OmegabNUFD2):(i+1) * len(OmegabNUFD2)]) for i in range(len(massNUFD2))]
MinimumChisqBBNNUFD2 = np.min(ChisqBBNNUFD2)
DeltaChisqBBNNUFD2 = MarginalizedChisqBBNNUFD2 - MinimumChisqBBNNUFD2

###########################################################################################################


# CMB
###########################################################################################################

ChisqCMBEEFD2 = (Omegab_fullEEFD2 - 0.02225) * ((Omegab_fullEEFD2 - 0.02225)*4.66287e7 + (NeffEEFD2 - 2.89)*-33110.3 + (HeEEFD2 - 0.246)*-496043.) + (NeffEEFD2 - 2.89) * ((Omegab_fullEEFD2 - 0.02225)*-33110.3 + (NeffEEFD2 - 2.89)*43.3733 + (HeEEFD2 - 0.246)*588.262) + (HeEEFD2 - 0.246)*((Omegab_fullEEFD2 - 0.02225)*-496043. + (NeffEEFD2 - 2.89)*588.262 + (HeEEFD2 - 0.246)*11168.2)
MarginalizedChisqCMBEEFD2 = [np.min(ChisqCMBEEFD2[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqCMBEEFD2 = np.min(ChisqCMBEEFD2)
DeltaChisqCMBEEFD2 = MarginalizedChisqCMBEEFD2 - MinimumChisqCMBEEFD2

ChisqCMBNUFD2 = (Omegab_fullNUFD2 - 0.02225) * ((Omegab_fullNUFD2 - 0.02225)*4.66287e7 + (NeffNUFD2 - 2.89)*-33110.3 + (HeNUFD2 - 0.246)*-496043.) + (NeffNUFD2 - 2.89) * ((Omegab_fullNUFD2 - 0.02225)*-33110.3 + (NeffNUFD2 - 2.89)*43.3733 + (HeNUFD2 - 0.246)*588.262) + (HeNUFD2 - 0.246)*((Omegab_fullNUFD2 - 0.02225)*-496043. + (NeffNUFD2 - 2.89)*588.262 + (HeNUFD2 - 0.246)*11168.2)
MarginalizedChisqCMBNUFD2 = [np.min(ChisqCMBNUFD2[i * len(OmegabNUFD2):(i+1) * len(OmegabNUFD2)]) for i in range(len(massNUFD2))]
MinimumChisqCMBNUFD2 = np.min(ChisqCMBNUFD2)
DeltaChisqCMBNUFD2 = MarginalizedChisqCMBNUFD2 - MinimumChisqCMBNUFD2


###########################################################################################################


# BBN + CMB
###########################################################################################################

ChisqBBNCMBEEFD2 = ChisqBBNEEFD2 + ChisqCMBEEFD2
MarginalizedChisqBBNCMBEEFD2 = [np.min(ChisqBBNCMBEEFD2[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqBBNCMBEEFD2 = np.min(ChisqBBNCMBEEFD2)
DeltaChisqBBNCMBEEFD2 = MarginalizedChisqBBNCMBEEFD2 - MinimumChisqBBNCMBEEFD2

ChisqBBNCMBNUFD2 = ChisqBBNNUFD2 + ChisqCMBNUFD2
MarginalizedChisqBBNCMBNUFD2 = [np.min(ChisqBBNCMBNUFD2[i * len(OmegabNUFD2):(i+1) * len(OmegabNUFD2)]) for i in range(len(massNUFD2))]
MinimumChisqBBNCMBNUFD2 = np.min(ChisqBBNCMBNUFD2)
DeltaChisqBBNCMBNUFD2 = MarginalizedChisqBBNCMBNUFD2 - MinimumChisqBBNCMBNUFD2

###########################################################################################################

print("Electrophilic Majorana:")
print("Minimum Chi squared BBN: {}".format(MinimumChisqBBNEEFD2))
print("Minimum Chi squared CMB: {}".format(MinimumChisqCMBEEFD2))
print("Minimum Chi squared BBN+CMB: {}".format(MinimumChisqBBNCMBEEFD2))
print("p-value BBN: {}".format(chi2.sf(MinimumChisqBBNEEFD2,1)))
print("p-value CMB: {}".format(chi2.sf(MinimumChisqCMBEEFD2,1)))
print("p-value BBN+CMB: {}".format(chi2.sf(MinimumChisqBBNCMBEEFD2,1)))

print("\nNeutrinophilic Majorana:")
print("Minimum Chi squared BBN: {}".format(MinimumChisqBBNNUFD2))
print("Minimum Chi squared CMB: {}".format(MinimumChisqCMBNUFD2))
print("Minimum Chi squared BBN+CMB: {}".format(MinimumChisqBBNCMBNUFD2))
print("p-value BBN: {}".format(chi2.sf(MinimumChisqBBNNUFD2,1)))
print("p-value CMB: {}".format(chi2.sf(MinimumChisqCMBNUFD2,1)))
print("p-value BBN+CMB: {}".format(chi2.sf(MinimumChisqBBNCMBNUFD2,1)))

stddev = np.array([0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856])
deltaChisq = []
for j in range(len(stddev)):
    deltaChisq.append(chi2.ppf(stddev[j], 1))
deltaChisq = np.sqrt(np.array(deltaChisq))

plt.figure()

plt.subplot(121)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.plot(massEEFD2, np.sqrt(DeltaChisqBBNEEFD2), linewidth=3, linestyle="dashdot", color="black", label="BBN")
plt.plot(massEEFD2, np.sqrt(DeltaChisqCMBEEFD2), linewidth=3, linestyle="dotted", color="black", label="CMB")
plt.plot(massEEFD2, np.sqrt(DeltaChisqBBNCMBEEFD2), linewidth=3, linestyle="solid", color="black", label="BBN+CMB")

for i in range(len(deltaChisq)):
    plt.axhline(deltaChisq[i])
    plt.annotate(r"{a:d}$\sigma$".format(a=i+1), (0.11, deltaChisq[i]+0.1), color="black", fontsize=20)

plt.axis(ymin=0, ymax=10, xmin=1e-1, xmax=14)
plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
plt.ylabel(r"$\sqrt{\Delta\tilde{\chi}^2}$", fontsize=20)
plt.semilogx()
plt.legend(loc='upper right', prop={'size': 16}, frameon=False, markerfirst=False)



plt.subplot(122)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.plot(massNUFD2, np.sqrt(DeltaChisqBBNNUFD2), linewidth=3, linestyle="dashdot", color="black", label="BBN")
plt.plot(massNUFD2, np.sqrt(DeltaChisqCMBNUFD2), linewidth=3, linestyle="dotted", color="black", label="CMB")
plt.plot(massNUFD2, np.sqrt(DeltaChisqBBNCMBNUFD2), linewidth=3, linestyle="solid", color="black", label="BBN+CMB")

for i in range(len(deltaChisq)):
    plt.axhline(deltaChisq[i])
    plt.annotate(r"{a:d}$\sigma$".format(a=i+1), (0.11, deltaChisq[i]+0.1), color="black", fontsize=20)

plt.axis(ymin=0, ymax=10, xmin=1e-1, xmax=14)
plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
plt.ylabel(r"$\sqrt{\Delta\tilde{\chi}^2}$", fontsize=20)
plt.semilogx()
plt.legend(loc='upper right', prop={'size': 16}, frameon=False, markerfirst=False)


plt.show()

###########################################################################################################

# typ = 3
# for sig, cl in enumerate(deltaChisq):
#     BBNInterp = UnivariateSpline(mass, DeltaChisqBBN-cl, s=0)
#     CMBInterp = UnivariateSpline(mass, DeltaChisqCMB-cl, s=0)
#     BBNCMBInterp = UnivariateSpline(mass, DeltaChisqBBNCMB-cl, s=0)
#     print("{}sigma:\t BBN = {}\t CMB = {}\t BBN+CMB = {}\n".format(sig+1,BBNInterp.roots(), CMBInterp.roots(), BBNCMBInterp.roots()))