# -*- coding: utf-8 -*-

import os
import itertools
import numpy
import argparse
import numpy as np

import subprocess

fiducial_params = [0.6736, 0.1200, 0.02237, 2.89, 0.246, 3.044, 0.9649, 0.0544]
h_dh = np.geomspace(1e-5, 1e-2, 10)
omc_dh = np.geomspace(1e-4, 1e-2, 10)
omb_dh = np.geomspace(1e-6, 1e-3, 10)
Neff_dh = np.geomspace(1e-3, 1e-1, 10)
YHe_dh = np.geomspace(1e-4, 1e-2, 10)
As_dh = np.geomspace(1e-4, 1e-1, 10)
ns_dh = np.geomspace(1e-5, 1e-2, 10)
tau_dh = np.geomspace(1e-5, 1e-3, 10)

h_dh = np.concatenate((-h_dh[::-1],[0.],h_dh))
omc_dh = np.concatenate((-omc_dh[::-1],[0.],omc_dh))
omb_dh = np.concatenate((-omb_dh[::-1],[0.],omb_dh))
Neff_dh = np.concatenate((-Neff_dh[::-1],[0.],Neff_dh))
YHe_dh = np.concatenate((-YHe_dh[::-1],[0.],YHe_dh))
As_dh = np.concatenate((-As_dh[::-1],[0.],As_dh))
ns_dh = np.concatenate((-ns_dh[::-1],[0.],ns_dh))
tau_dh = np.concatenate((-tau_dh[::-1],[0.],tau_dh))


spectra_h = []
spectra_omcdm = []
spectra_omb = []
spectra_Neff = []
spectra_YHe = []
spectra_As = []
spectra_ns = []
spectra_tau = []


for i in range(len(fiducial_params)):
    for j in range(10):
        subprocess.call(["../class_public/.class", "../class_public/spectra.ini"])
        # while True:
        #     try:
        #         subprocess.run(["./class", "relvel.ini"], timeout=60)
        #         break
        #     except:
        #         pass
        spectra = np.loadtxt("output/relvel_z1_tk.dat", skiprows=1, unpack=True)

        wavenumbers = z_10[0]
        if first == True:
            save = np.zeros((4, 11, len(wavenumbers)))
            first = False
