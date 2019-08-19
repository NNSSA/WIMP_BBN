# -*- coding: utf-8 -*-

import numpy as np
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

multipole_min = 30
multipole_max = 5000
multipoles = np.arange(multipole_min, multipole_max+1, 1)
position_fiducial = 10

derivatives = dict(zip(params_names, [{key:None for key in spectra_names} for i in range(len_params)]))

for name, spectrum in spectra.items():
    for key in spectra_names:

        temp = [[] for num in range((len(dh[name])))]

        for i in range(len(dh[name])):
            temp[i] = (spectrum[key][::-1][i] - spectrum[key][i]) / (params[name][::-1][i] - params[name][i])

        derivatives[name][key] = temp

################################################
# Plot and save the derivatives
################################################

for name, derivs in derivatives.items():
    plt.figure(figsize=(14,20))

    for counter, key in enumerate(derivs.keys()):
        plt.subplot(4,1, counter+1)
        parameters = np.arange(0, len(dh[name]), 1)
        norm = matplotlib.colors.LogNorm(
                vmin=2.*np.min(dh[name]),
                vmax=2.*np.max(dh[name])
        )
        c_m = matplotlib.cm.PiYG
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        for parameter in parameters[::-1]:
            plt.plot(multipoles, derivs[key][parameter]*1e10 * multipoles * (multipoles + 1) / 2. / np.pi, linewidth=2, color=s_m.to_rgba(dh[name][parameter]))

        cb = plt.colorbar(s_m)
        cb.set_label(label=r'$\Delta {}$'.format(name), fontsize=16, rotation=270, labelpad=20)
        cb.ax.tick_params(labelsize=16)

        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=16)
        plt.axis(xmin=multipole_min, xmax=multipole_max)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.ylabel(r"$10^{{10}}\times\frac{{\ell(\ell+1)}}{{2\pi}}\times \frac{\mathrm{d}" + r"{}".format(key) + r"}{\mathrm{d}" + r"{}".format(name) + r"}$", fontsize=20)

    plt.savefig("derivatives/derivatives_{}".format(name))

# plt.show()
