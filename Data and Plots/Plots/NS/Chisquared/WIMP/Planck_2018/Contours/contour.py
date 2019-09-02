import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter

dataEEFD2 = np.loadtxt("../../Case=EE_Stat=FD_gDM=2._Sigmav=0._BR=0", skiprows=1)
massEEFD2 = np.unique(dataEEFD2[:,0])
Omegab_fullEEFD2 = dataEEFD2[:,1]
OmegabEEFD2 = np.unique(Omegab_fullEEFD2)
NeffEEFD2 = dataEEFD2[:,2]
HeEEFD2 = dataEEFD2[:,8]*4
DoverHEEFD2 = dataEEFD2[:,5]/dataEEFD2[:,4]

# BBN
###########################################################################################################

sigmasqHe = (0.003)**2 + (0.00017)**2
sigmasqDoverH = (0.027e-5)**2 + (0.036e-5)**2

MeasuredHe = 0.245
MeasuredDoverH = 2.569e-5

ChisqBBN = (HeEEFD2 - MeasuredHe)**2 / sigmasqHe + (DoverHEEFD2 - MeasuredDoverH)**2 / sigmasqDoverH
x = massEEFD2
y = OmegabEEFD2
XX, YY = np.meshgrid(x,y)
ZZBBN = np.transpose(np.reshape(ChisqBBN, (len(x), len(y))))
ZZBNN = gaussian_filter(ZZBBN, 0.3)

###########################################################################################################


# CMB
###########################################################################################################

ChisqCMB = (Omegab_fullEEFD2 - 0.02225) * ((Omegab_fullEEFD2 - 0.02225)*4.66287e7 + (NeffEEFD2 - 2.89)*-33110.3 + (HeEEFD2 - 0.246)*-496043.) + (NeffEEFD2 - 2.89) * ((Omegab_fullEEFD2 - 0.02225)*-33110.3 + (NeffEEFD2 - 2.89)*43.3733 + (HeEEFD2 - 0.246)*588.262) + (HeEEFD2 - 0.246)*((Omegab_fullEEFD2 - 0.02225)*-496043. + (NeffEEFD2 - 2.89)*588.262 + (HeEEFD2 - 0.246)*11168.2)
MarginalizedChisqCMB = [np.min(ChisqCMB[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqCMB = np.min(ChisqCMB)
DeltaChisqCMB = MarginalizedChisqCMB - MinimumChisqCMB
ZZCMB = np.transpose(np.reshape(ChisqCMB, (len(x), len(y))))

###########################################################################################################


# BBN + CMB
###########################################################################################################

ChisqBBNCMB = ChisqBBN + ChisqCMB
MarginalizedChisqBBNCMB = [np.min(ChisqBBNCMB[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqBBNCMB = np.min(ChisqBBNCMB)
DeltaChisqBBNCMB = MarginalizedChisqBBNCMB - MinimumChisqBBNCMB
ZZBBNCMB = np.transpose(np.reshape(ChisqBBNCMB, (len(x), len(y))))

###########################################################################################################


# CMB S-4
###########################################################################################################
Cov_planck_s4 = np.loadtxt("Covariance_Planck_S4.txt")
Fish_planck_s4 = np.linalg.inv(Cov_planck_s4)

OmbM = 0.02237
NeffM = 3.046
HeM = 0.246

ChisqS4 = (Omegab_fullEEFD2 - OmbM) * ((Omegab_fullEEFD2 - OmbM)*Fish_planck_s4[0,0] + (NeffEEFD2 - NeffM)*Fish_planck_s4[0,1] + (HeEEFD2 - HeM)*Fish_planck_s4[0,2]) + (NeffEEFD2 - NeffM) * ((Omegab_fullEEFD2 - OmbM)*Fish_planck_s4[1,0] + (NeffEEFD2 - NeffM)*Fish_planck_s4[1,1] + (HeEEFD2 - HeM)*Fish_planck_s4[1,2]) + (HeEEFD2 - HeM)*((Omegab_fullEEFD2 - OmbM)*Fish_planck_s4[2,0] + (NeffEEFD2 - NeffM)*Fish_planck_s4[2,1] + (HeEEFD2 - HeM)*Fish_planck_s4[2,2])
MarginalizedChisqS4 = [np.min(ChisqS4[i * len(OmegabEEFD2):(i+1) * len(OmegabEEFD2)]) for i in range(len(massEEFD2))]
MinimumChisqS4 = np.min(ChisqS4)
DeltaChisqS4 = MarginalizedChisqS4 - MinimumChisqS4
ZZS4 = np.transpose(np.reshape(ChisqS4, (len(x), len(y))))

###########################################################################################################


# stddev = np.array([0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856])
stddev = np.array([0., 0.954499736103642, 0.997300203936740])
deltaChisq = []
for j in range(len(stddev)):
    deltaChisq.append(chi2.ppf(stddev[j], 2))

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

CS = plt.contourf(XX, YY, ZZBBN, linewidths=3, cmap=plt.get_cmap('summer'),vmin=7,vmax=10,
             levels=[deltaChisq[0], deltaChisq[1], deltaChisq[2]],#, deltaChisq[2], deltaChisq[3], deltaChisq[4]],
             interpolation="nearest")
CS2 = plt.contourf(XX, YY, ZZCMB, linewidths=3, cmap=plt.get_cmap('autumn'),vmin=7,vmax=10,
             levels=[deltaChisq[0], deltaChisq[1], deltaChisq[2]],#, deltaChisq[2], deltaChisq[3], deltaChisq[4]],
             interpolation="nearest")
CS3 = plt.contourf(XX, YY, ZZBBNCMB, linewidths=3, cmap=plt.get_cmap('winter'),vmin=7,vmax=10,
             levels=[deltaChisq[0], deltaChisq[1], deltaChisq[2]],#, deltaChisq[3], deltaChisq[4]],
             interpolation="nearest")
CS3 = plt.contour(XX, YY, ZZS4, linewidths=3, cmap=plt.get_cmap('binary'),vmin=9,vmax=10,
             levels=[deltaChisq[1], deltaChisq[2]],#, deltaChisq[3], deltaChisq[4]],
             interpolation="nearest")
# fmt = {}
# # strs = [r'$1\sigma$', r'$2\sigma$',r'$3\sigma$',r'$4\sigma$',r'$5\sigma$']
# strs = [r'$2\sigma$', r'$3\sigma$']
# for l, s in zip(CS.levels, strs):
#     fmt[l] = s
# plt.clabel(CS, colors="black", fmt=fmt, inline=True, inline_spacing=20, fontsize=18)#, manual=True)

plt.annotate("BBN", (10,0.0213), color="black", fontsize=25)
plt.annotate("CMB", (5,0.022), color="black", fontsize=25)
plt.annotate("BBN+CMB", (12,0.02205), color="black", fontsize=25)
plt.annotate("CMB-S4", (25,0.02219), color="black", fontsize=25)
plt.axis(ymin=0.02, ymax=0.023, xmin=0.4, xmax=30)
plt.xlabel(r"$m_\chi$ [MeV]", fontsize=20)
plt.ylabel(r"$\Omega_\mathrm{b}h^2$", fontsize=20)
plt.show()
