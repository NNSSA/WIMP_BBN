# -*- coding: utf-8 -*-

import numpy as np
import subprocess
import pickle
import pprint

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
# Call CLASS and store spectra
################################################

lmin_class = 2
lmin = 30
lmax_temp = 3000
lmax_pol = 5000
index_low = lmin-lmin_class
index_up_temp = lmax_temp - lmin_class + 1
index_up_pol = lmax_pol - lmin_class + 1

spectra = dict(zip(params_names, [{key:[] for key in spectra_names} for i in range(len_params)]))
spectra_unlensed = dict(zip(params_names, [{key:[] for key in spectra_names} for i in range(len_params)]))

for name, value in params.items():
    for j in range(len(value)):

        theta = params[name][j] if name == r"100\theta_\mathrm{s}" else LCDM_params[r"100\theta_\mathrm{s}"]
        omc = params[name][j] if name == r"\Omega_\mathrm{c}h^2" else LCDM_params[r"\Omega_\mathrm{c}h^2"]
        omb = params[name][j] if name == r"\Omega_\mathrm{b}h^2" else LCDM_params[r"\Omega_\mathrm{b}h^2"]
        Neff = params[name][j] if name == r"N_\mathrm{eff}" else LCDM_params[r"N_\mathrm{eff}"]
        YHe = params[name][j] if name == r"Y_\mathrm{He}" else LCDM_params[r"Y_\mathrm{He}"]
        As = params[name][j] if name == r"\ln(10^{10}As)" else LCDM_params[r"\ln(10^{10}As)"]
        ns = params[name][j] if name == r"n_\mathrm{s}" else LCDM_params[r"n_\mathrm{s}"]
        tau = params[name][j] if name == r"\tau" else LCDM_params[r"\tau"]


        with open("spectra_base.ini") as infile, open("../../../spectra.ini", "w") as outfile:
            for i,line in enumerate(infile):
                if i==0:
                    outfile.write("100*theta_s = {}\n".format(theta))
                elif i==1:
                    outfile.write("omega_cdm = {}\n".format(omc))
                elif i==2:
                    outfile.write("omega_b = {}\n".format(omb))
                elif i==3:
                    outfile.write("N_eff = {}\n".format(Neff))
                elif i==4:
                    outfile.write("YHe = {}\n".format(YHe))
                elif i==5:
                    outfile.write("ln10^{10}A_s = " + "{}\n".format(As))
                elif i==6:
                    outfile.write("n_s = {}\n".format(ns))
                elif i==7:
                    outfile.write("tau_reio = {}\n".format(tau))
                else:
                    outfile.write(line)

        print("\n", theta, omc, omb, Neff, YHe, As, ns, tau, "\n")
        subprocess.call(["./class", "../spectra.ini", "cl_ref.pre"], cwd="../../../class")

        temp = np.loadtxt("../../../class/output/spectracl_lensed.dat", skiprows=11, unpack=True)
        prefactor = 2. * np.pi / (temp[0] * (temp[0] + 1.)) # to get the pure Cl's
        spectra[name][spectra_names[0]].append(
            np.concatenate(
                (
                temp[1][index_low:index_up_temp] * prefactor[index_low:index_up_temp],
                np.zeros(lmax_pol - lmax_temp)
                )
            )
        )
        spectra[name][spectra_names[1]].append(temp[2][index_low:index_up_pol] * prefactor[index_low:index_up_pol])
        spectra[name][spectra_names[2]].append(temp[3][index_low:index_up_pol] * prefactor[index_low:index_up_pol])
        spectra[name][spectra_names[3]].append(temp[4][index_low:index_up_pol] * prefactor[index_low:index_up_pol])



        temp_unlensed = np.loadtxt("../../../class/output/spectracl.dat", skiprows=11, unpack=True)
        prefactor_unlensed = 2. * np.pi / (temp_unlensed[0] * (temp_unlensed[0] + 1.)) # to get the pure Cl's
        spectra_unlensed[name][spectra_names[0]].append(
            np.concatenate(
                (
                temp_unlensed[1][index_low:index_up_temp] * prefactor_unlensed[index_low:index_up_temp],
                np.zeros(lmax_pol - lmax_temp)
                )
            )
        )
        spectra_unlensed[name][spectra_names[1]].append(temp_unlensed[2][index_low:index_up_pol] * prefactor_unlensed[index_low:index_up_pol])
        spectra_unlensed[name][spectra_names[2]].append(temp_unlensed[3][index_low:index_up_pol] * prefactor_unlensed[index_low:index_up_pol])
        spectra_unlensed[name][spectra_names[3]].append(temp_unlensed[4][index_low:index_up_pol] * prefactor_unlensed[index_low:index_up_pol])


################################################
# Output spectra
################################################

forexport = {name : open("spectra/spectra_{}_lensed.pkl".format(name), "wb") for name in params_names}
forexport_unlensed = {name : open("spectra/spectra_{}_unlensed.pkl".format(name), "wb") for name in params_names}

for name in params_names:
    pickle.dump(spectra[name], forexport[name])
    pickle.dump(spectra_unlensed[name], forexport_unlensed[name])
    forexport[name].close()
    forexport_unlensed[name].close()

