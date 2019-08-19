import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2

data = np.loadtxt("../Case=EE_Stat=FD_gDM=2._Sigmav=0._BR=0", skiprows=1)
mass = data[:,0][::41]
Omegab_full = data[:,1]
Omegab = data[:41,1]
Neff = data[:,2]
He = data[:,8]*4
DoverH = data[:,5]/data[:,4]


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

Cov_planck_s4 = np.loadtxt("Covariance_Planck_S4.txt")
Fish_planck_s4 = np.linalg.inv(Cov_planck_s4)

ChisqCMB = (Omegab_full - 0.02225) * ((Omegab_full - 0.02225)*Fish_planck_s4[0,0] + (Neff - 2.89)*Fish_planck_s4[0,1] + (He - 0.246)*Fish_planck_s4[0,2]) + (Neff - 2.89) * ((Omegab_full - 0.02225)*Fish_planck_s4[1,0] + (Neff - 2.89)*Fish_planck_s4[1,1] + (He - 0.246)*Fish_planck_s4[1,2]) + (He - 0.246)*((Omegab_full - 0.02225)*Fish_planck_s4[2,0] + (Neff - 2.89)*Fish_planck_s4[2,1] + (He - 0.246)*Fish_planck_s4[2,2])
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
print("Minimum Chi squared BBN: {}".format(MinimumChisqBBN))
print("Minimum Chi squared CMB: {}".format(MinimumChisqCMB))
print("Minimum Chi squared BBN+CMB: {}".format(MinimumChisqBBNCMB))
print("p-value BBN: {}".format(chi2.sf(MinimumChisqBBN,1)))
print("p-value CMB: {}".format(chi2.sf(MinimumChisqCMB,1)))
print("p-value BBN+CMB: {}".format(chi2.sf(MinimumChisqBBNCMB,1)))

stddev = np.array([0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856])
deltaChisq = []
for j in range(len(stddev)):
    deltaChisq.append(chi2.ppf(stddev[j], 1))

plt.figure()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.plot(mass, DeltaChisqBBN, linewidth=3, linestyle="dashdot", color="black", label="BBN")
plt.plot(mass, DeltaChisqCMB, linewidth=3, linestyle="dotted", color="black", label="CMB")
plt.plot(mass, DeltaChisqBBNCMB, linewidth=3, linestyle="solid", color="black", label="BBN+CMB")
for i in range(len(deltaChisq)):
    plt.axhline(deltaChisq[i])
    plt.annotate(r"{a:d}$\sigma$".format(a=i+1), (0.11, deltaChisq[i]+0.2), color="black", fontsize=20)

plt.axis(ymin=0, ymax=80, xmin=1e-1, xmax=14)
plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
plt.ylabel(r"$\Delta\tilde{\chi}^2$", fontsize=20)
plt.semilogx()
plt.legend(loc='upper right', prop={'size': 16}, frameon=False, markerfirst=False)
plt.show()
