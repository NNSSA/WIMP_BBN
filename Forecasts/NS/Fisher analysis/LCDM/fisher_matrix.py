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
    r"100\theta_\mathrm{s}":1.04112,
    r"\Omega_\mathrm{c}h^2":0.1188,
    r"\Omega_\mathrm{b}h^2":0.02230,
    r"\ln(10^{10}As)":3.064,
    r"n_\mathrm{s}":0.9667,
    r"\tau":0.066
}

dh = {
    r"100\theta_\mathrm{s}":np.geomspace(1e-7, 2e-3, 10),
    r"\Omega_\mathrm{c}h^2":np.geomspace(1e-6, 2e-3, 10),
    r"\Omega_\mathrm{b}h^2":np.geomspace(1e-7, 8e-4, 10),
    r"\ln(10^{10}As)":np.geomspace(1e-5, 5e-2, 10),
    r"n_\mathrm{s}":np.geomspace(1e-6, 1e-2, 10),
    r"\tau":np.geomspace(1e-5, 2e-2, 10)
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
point = 1
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

cov_planck = np.loadtxt("base_plikHM_TTTEEE_lowl_lowE_lensing.covmat", skiprows=1)
fisher_planck = np.linalg.inv(cov_planck)
planck_pos = [2, 1, 0, 4, 5, 3]
planck_pos_sort = np.argsort(planck_pos)
fisher_planck_ordered = [[fisher_planck[key1, key2] for key1 in planck_pos_sort] for key2 in planck_pos_sort]

fisher_planck_stage4 = fisher_planck_ordered + fisher_matrix
covariance_planck_stage4 = np.linalg.inv(fisher_planck_stage4)


################################################
# Print Fisher matrix
################################################

print("\nStage-IV standard deviations:")
for num in range(len_params):
    print("{}: {}".format(params_names[num], np.sqrt(inverse_fisher_matrix[num,num])))


print("\nPlanck2018 + Stage-IV standard deviations:")
for num in range(len_params):
    print("{}: {}".format(params_names[num], np.sqrt(covariance_planck_stage4[num,num])))
