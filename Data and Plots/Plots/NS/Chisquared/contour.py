import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter

data = np.loadtxt("Neutrinophillic_Boson_2_dof.txt", skiprows=1)
mass = data[:,0][::41]
Omegab_full = data[:,1]
Omegab = data[:41,1]
Neff = data[:,2]
He = data[:,8]*4
DoverH = data[:,5]/data[:,4]

sigmasqHe = (0.003)**2 + (0.00017)**2
sigmasqDoverH = (0.027e-5)**2 + (0.036e-5)**2

MeasuredHe = 0.245
MeasuredDoverH = 2.569e-5

# BBN
###########################################################################################################

ChisqBBN = (He - MeasuredHe)**2 / sigmasqHe + (DoverH - MeasuredDoverH)**2 / sigmasqDoverH
x = mass
y = Omegab
XX, YY = np.meshgrid(x,y)
ZZ = np.transpose(np.reshape(ChisqBBN, (len(x), len(y))))
ZZ = gaussian_filter(ZZ, 0.3)

###########################################################################################################


# CMB
###########################################################################################################

rho12 = 0.4
rho13 = 0.18
rho23 = -0.69

SigmaCMB = 0.00022 * (0.00022 + rho12 * 0.31 + rho13 * 0.018) + 0.31 * (rho12 * 0.00022 + 0.31 + rho23 * 0.018) +  0.018 * (rho13 * 0.00022 + rho23 * 0.31 + 0.018)
ChisqCMB = ((Omegab_full - 0.00223)**2 + (Neff - 2.89)**2 + (He - 0.246)**2) / SigmaCMB
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


stddev = np.array([0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856])
deltaChisq = []
for j in range(len(stddev)): 
    deltaChisq.append(chi2.ppf(stddev[j], 2))

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

CS = plt.contour(XX, YY, ZZ, linewidths=3, cmap=plt.get_cmap('summer'),vmin=0,vmax=50,
             levels=[deltaChisq[0], deltaChisq[1], deltaChisq[2], deltaChisq[3], deltaChisq[4]], 
             interpolation="nearest")  
fmt = {}
strs = [r'$1\sigma$', r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$']
for l, s in zip(CS.levels, strs):
    fmt[l] = s
plt.clabel(CS, colors="black", fmt=fmt, inline=True, inline_spacing=20, fontsize=18, manual=True)

plt.axis(ymin=0.02, ymax=0.025, xmin=0.4, xmax=30)
plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
plt.ylabel(r"$\Omega_\mathrm{b}h^2$", fontsize=20)
plt.show()
