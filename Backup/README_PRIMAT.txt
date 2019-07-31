%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This is the README file for PRIMAT (second version, 0.1.1), a code to perform BBN computations, written and maintained by Cyril Pitrou and Alain Coc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
10 septembre 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-The main code of PRIMAT is the file PRIMAT-Main.nb
First open the notebook in Mathematica. Then in the Evaluation menu, select ‘Evaluate notebook’. If asked the question ‘evaluate initialization cells first ?’, answer no. 
Then Mathematica proceeds in evaluating all cells, in order, and it should reach the end of the notebook (with the plots and results) in less than one minute.


-Furthermore, when PRIMAT-Main.nb is opened and saved in Mathematica, the cells which are 'initialization cells' are saved into PRIMAT-Main.m. 
This file contains then all the principal definitions and functions and variables
which can be loaded from another notebook to perform BBN computations and analysis of results. The ‘Examples’ Folder contains several typical applications which work exactly like that (first it loads all necessary definitions stored in PRIMAT-Main.m and then it performs a few useful computations for each selected example).


-PRIMAT_small_network_old.nb is a modified version of PRIMAT.nb which uses only the minimum set of nuclear reaction (11 + weak-interactions) and nuclides (neutron, H1 H2 H3 He3 He4 Li7 Be7) which is much faster to solve numerically. It loads the reduced set of reactions stored in BBNRates_2018_11reactions.dat and uses no analytic reaction. However it is still possible to add a few analytic reactions so as to study their effect on the reduced network (see dedicated section of the notebook).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*’Examples’ Folder

Several examples are located in the Examples folder. The most important ones are

1) PRIMAT-Abundances-and-Plots1.nb -> Basic abundances computation and nice plots of abundance evolution during BBN

2) PRIMAT-Abundances-Eta1.nb -> Abundances as functions of baryon to photon ratio eta, and associated plots.

3) PRIMAT-AbundanceFit1.nb -> Fit of abundances as a polynomial in baryons abundance, neutrino generations, and neutron life time.

4) PRIMAT-Omegab-Neff-For-CAMB.nb -> exports abundances of He4 and deuterium depending on a grid of baryon abundance and neutrino generations. The file generated was used to generate the table provided for CAMB.


If you use PRIMAT for publications, please cite along the website and the companion paper (Pitrou et al. 2018, arXiv:1801.08023).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*’Interpolations’ Folder
Interpolations of weak rates (but also for QED plasma corrections) are computed and stored in this folder. They can be overwritten if appropriate booleans in PRIMAT-Main.nb are set to True ($RecomputeWeakRates,$RecomputePlasmaCorrections). On some machines and for some versions of Mathematica, there is an incompatibility with the files which are packaged in the PRIMAT zip file. In that case, the code outputs non-sense as it cannot read them. Just erase all files in the ‘Interpolations’ folder to force Mathematica recomputing them. The first time, it might take a few minutes or about an hour and then the second time it should run rapidly as it would use the newly created files.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Should you find difficulties in using PRIMAT, please email to
pitrou@iap.fr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The file PRIMAT-Main.pdf is a pdf dump of the Mathematica notebook
that can be used to visualize the code without a Mathematica front-end.

The file PhysReptArxivVersion.pdf is the arXiv version (1801.08023) of the published Physics Reports, 04, (2018) 005. All equations in the code refer to the arXiv version numbering.
