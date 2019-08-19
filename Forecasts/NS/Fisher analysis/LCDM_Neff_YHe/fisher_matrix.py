# -*- coding: utf-8 -*-

import numpy as np
import itertools
import subprocess
import pickle
import pprint
import matplotlib
import matplotlib.pyplot as plt

################################################
# Define fiducial parameters and deviations
################################################

LCDM_params = {
    r"100\theta_\mathrm{s}":1.04092,
    r"\Omega_\mathrm{c}h^2":0.1200,
    r"\Omega_\mathrm{b}h^2":0.02237,
    r"N_\mathrm{eff}":3.046,
    r"Y_\mathrm{He}":0.246,
    r"\ln(10^{10}As)":3.044,
    r"n_\mathrm{s}":0.9649,
    r"\tau":0.0544
}

dh = {
    r"100\theta_\mathrm{s}":np.geomspace(1e-7, 1e-4, 10),
    r"\Omega_\mathrm{c}h^2":np.geomspace(1e-6, 1e-3, 10),
    r"\Omega_\mathrm{b}h^2":np.geomspace(1e-7, 1e-4, 10),
    r"N_\mathrm{eff}":np.geomspace(1e-4, 3e-1, 10),
    r"Y_\mathrm{He}":np.geomspace(1e-5, 1e-3, 10),
    r"\ln(10^{10}As)":np.geomspace(1e-5, 1e-2, 10),
    r"n_\mathrm{s}":np.geomspace(1e-6, 1e-3, 10),
    r"\tau":np.geomspace(1e-5, 1e-3, 10)
}

spectra_names = ["C_\ell^{TT}", "C_\ell^{EE}", "C_\ell^{TE}", "C_\ell^{BB}"]
params_names = list(LCDM_params.keys())
len_params = len(params_names)

################################################
# Compute range of parameters
################################################

params = dict()

for name, value in LCDM_params.items():
    params[name] = value + np.concatenate((-dh[name][::-1], [0.], dh[name]))

################################################
# Import spectra
################################################

spectra = dict(zip(params_names, [{key:[] for key in spectra_names} for i in range(len_params)]))

for name in params_names:
    forimport = open("spectra/spectra_{}_lensed.pkl".format(name), 'rb')
    spectra[name] = pickle.load(forimport)
    forimport.close()

################################################
# Compute symmetric derivatives
################################################

lmin = 30
lmax = 5000
multipoles = np.arange(lmin, lmax+1, 1)
position_fiducial = 10
point = 5
derivatives = dict(zip(params_names, [{key:None for key in spectra_names} for i in range(len_params)]))

for name, spectrum in spectra.items():
    for key in spectra_names:
        derivatives[name][key] = (spectrum[key][position_fiducial+point] - spectrum[key][position_fiducial-point]) \
            / (params[name][position_fiducial+point] - params[name][position_fiducial-point])

################################################
# Compute Fisher matrix and its inverse
################################################

fidspectra = {key:spectra[params_names[0]][key][position_fiducial] for key in spectra_names}

TT = spectra_names[0]
EE = spectra_names[1]
TE = spectra_names[2]
BB = spectra_names[3]

theta_FWHM = 2. # in arcminutes
theta_FWHM *= np.pi / 180. / 60. # in radians
fsky = 0.4
T_CMB = 2.7255e6 # μK

def noise(name, l):
    temp = np.exp(l*(l+1) * theta_FWHM**2 / (8. * np.log(2))) * ((np.pi / 180. / 60.)**2) / T_CMB**2
    if name == TE:
        temp *= 0.
    elif name == TT:
        temp *= 1.
    elif name in [EE, BB]:
        temp *= 2.
    return temp


def covariance(l):
    assert l >= lmin, "l < l_min"
    spectrumTT = fidspectra[TT][l-lmin] + noise(TT, l)
    spectrumEE = fidspectra[EE][l-lmin] + noise(EE, l)
    spectrumTE = fidspectra[TE][l-lmin] + noise(TE, l)
    spectrumBB = fidspectra[BB][l-lmin] + noise(BB, l)

    matrix = (2. / ((2.*l + 1.)*fsky))  * np.array([
        [(spectrumTT)**2, (spectrumTE)**2, spectrumTT*spectrumTE, 0.],
        [(spectrumTE)**2, (spectrumEE)**2, spectrumEE*spectrumTE, 0.],
        [spectrumTT*spectrumTE, spectrumEE*spectrumTE, 0.5 * ((spectrumTE)**2 + spectrumTT*spectrumEE), 0.],
        [0., 0., 0., (spectrumBB)**2]
    ])
    return np.linalg.inv(matrix)

fisher = dict(zip(list(itertools.product(params_names, repeat=2)), [0. for i in range(len(params_names)**2)]))

for param1, param2 in list(itertools.product(params_names, repeat=2)):
    for ell in multipoles:
        deriv1 = np.array(list(derivatives[param1].values()))[:,ell-lmin]
        deriv2 = np.array(list(derivatives[param2].values()))[:,ell-lmin]
        fisher[param1, param2] += deriv1 @ (covariance(ell) @ deriv2)

fisher_matrix = np.array([[fisher[(key1, key2)] for key2 in params_names] for key1 in params_names])
inverse_fisher_matrix = np.linalg.inv(fisher_matrix)

positions = [2, 3, 4]
slice_fisher_matrix = np.array([[fisher_matrix[key1, key2] for key2 in positions] for key1 in positions])
inverse_slice_fisher_matrix = np.array([[inverse_fisher_matrix[key1, key2] for key2 in positions] for key1 in positions])

cov_planck = np.loadtxt("base_nnu_yhe_plikHM_TTTEEE_lowl_lowE.covmat", skiprows=1)
fisher_planck = np.linalg.inv(cov_planck)
planck_pos = [2, 1, 0, 7, 3, 4, 5, 6]
planck_pos_sort = np.argsort(planck_pos)
fisher_planck_ordered = [[fisher_planck[key1, key2] for key1 in planck_pos_sort] for key2 in planck_pos_sort]

fisher_planck_stage4 = fisher_planck_ordered + fisher_matrix
covariance_planck_stage4 = np.linalg.inv(fisher_planck_stage4) # Planck 2018 + Stage-IV (Omegabh^2, Neff, YHe)

slice_fisher_planck_stage4 = np.array([[fisher_planck_stage4[key1, key2] for key2 in positions] for key1 in positions])
slice_covariance_planck_stage4 = np.array([[covariance_planck_stage4[key1, key2] for key2 in positions] for key1 in positions])


################################################
# Print Fisher matrix
################################################

# pprint.pprint(fisher)
print("\nTotal Stage-IV Fisher matrix:\n", params_names, "\n", fisher_matrix, "\n")
print("\nTotal Stage-IV covariance matrix:\n", inverse_fisher_matrix, "\n")
print("\nFisher x covariance", fisher_matrix@inverse_fisher_matrix, "\n")

print("\nStage-IV Fisher matrix (Omegabh^2, Neff, YHe):\n", slice_fisher_matrix, "\n")
print("\nStage-IV covariance matrix (Omegabh^2, Neff, YHe):\n", inverse_slice_fisher_matrix, "\n")

print("\nStage-IV standard deviations:")
for num in range(len_params):
    print("{}: {}".format(params_names[num], np.sqrt(inverse_fisher_matrix[num,num])))


print("\nPlanck2018 + Stage-IV Fisher matrix:\n", params_names, "\n", fisher_planck_stage4, "\n")
print("\nPlanck2018 + Stage-IV covariance matrix:\n", covariance_planck_stage4, "\n")
print("\nFisher x covariance", fisher_planck_stage4@covariance_planck_stage4, "\n")

print("\nPlanck2018 + Stage-IV Fisher matrix (Omegabh^2, Neff, YHe):\n", slice_fisher_planck_stage4, "\n")
print("\nPlanck2018 + Stage-IV covariance matrix (Omegabh^2, Neff, YHe):\n", slice_covariance_planck_stage4, "\n")

print("\nPlanck2018 + Stage-IV standard deviations:")
for num in range(len_params):
    print("{}: {}".format(params_names[num], np.sqrt(covariance_planck_stage4[num,num])))


np.savetxt("Covariance_Planck_S4.txt", slice_covariance_planck_stage4)