import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.interpolate import interp1d

data = np.loadtxt("SBBN.txt", skiprows=1)
Omegab = data[:,0]
Neff = data[:,1]
He = data[:,7]*4
DoverH = data[:,4]/data[:,3]

sigmasqHe = (0.003)**2 + (0.00017)**2
sigmasqDoverH = (0.027e-5)**2 + (0.036e-5)**2

MeasuredHe = 0.245
MeasuredDoverH = 2.569e-5

# BBN
###########################################################################################################

ChisqBBN = (He - MeasuredHe)**2 / sigmasqHe + (DoverH - MeasuredDoverH)**2 / sigmasqDoverH
MinimumChisqBBN = np.min(ChisqBBN)
DeltaChisqBBN = ChisqBBN - MinimumChisqBBN
rangexBBN = np.linspace(0.02,0.024,1000)
rangeyBBN = interp1d(Omegab, DeltaChisqBBN, kind="quadratic")(rangexBBN)

###########################################################################################################


# CMB
###########################################################################################################

ChisqCMB = (Omegab - 0.02225) * ((Omegab - 0.02225)*4.66287e7 + (Neff - 2.89)*-33110.3 + (He - 0.246)*-496043.) + (Neff - 2.89) * ((Omegab - 0.02225)*-33110.3 + (Neff - 2.89)*43.3733 + (He - 0.246)*588.262) + (He - 0.246)*((Omegab - 0.02225)*-496043. + (Neff - 2.89)*588.262 + (He - 0.246)*11168.2)
MinimumChisqCMB = np.min(ChisqCMB)
DeltaChisqCMB = ChisqCMB - MinimumChisqCMB
rangexCMB = np.linspace(0.02,0.024,1000)
rangeyCMB = interp1d(Omegab, DeltaChisqCMB, kind="quadratic")(rangexCMB)

###########################################################################################################


# BBN + CMB
###########################################################################################################

ChisqBBNCMB = ChisqBBN + ChisqCMB
MinimumChisqBBNCMB = np.min(ChisqBBNCMB)
DeltaChisqBBNCMB = ChisqBBNCMB - MinimumChisqBBNCMB
rangexBBNCMB = np.linspace(0.02,0.024,1000)
rangeyBBNCMB = interp1d(Omegab, DeltaChisqBBNCMB, kind="quadratic")(rangexBBNCMB)

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

#plt.plot(rangexBBN, rangeyBBN, linewidth=3, linestyle="dashdot", color="black", label="BBN")
plt.plot(Omegab, DeltaChisqBBN, linewidth=3, linestyle="dashdot", color="black", label="BBN")
#plt.plot(rangexCMB, rangeyCMB, linewidth=3, linestyle="dotted", color="black", label="CMB")
plt.plot(Omegab, DeltaChisqCMB, linewidth=3, linestyle="dotted", color="black", label="CMB")
#plt.plot(rangexBBNCMB, rangeyBBNCMB, linewidth=3, linestyle="solid", color="black", label="BBN+CMB")
plt.plot(Omegab, DeltaChisqBBNCMB, linewidth=3, linestyle="solid", color="black", label="BBN+CMB")
for i in range(len(deltaChisq)):
    plt.axhline(deltaChisq[i], linestyle="solid")
    plt.annotate(r"{a:d}$\sigma$".format(a=i+1), (0.02005, deltaChisq[i]+0.3*(0.3*i+1)), color="black", fontsize=20)

plt.axis(ymin=0, ymax=40, xmin=0.02, xmax=0.024)
plt.xlabel(r"$\Omega_\mathrm{b}h^2$", fontsize=20)
plt.ylabel(r"$\Delta\tilde{\chi}^2$", fontsize=20)
plt.legend(loc='upper right', prop={'size': 16}, frameon=False, markerfirst=False)
#plt.semilogy()
plt.show()
