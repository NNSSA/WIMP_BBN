Changes to PRIMI
----------------

* $EvolutionType flag to specify SM, Neff, or WIMP evolution
* Parameters in the evolution must be specified outside the code e.g. mDM, gDM etc.
* TnuStart defined with PrimiCosmo
* BBN options moved to PrimiNuc
* Baryon definitions moved to PrimiNuc
* Removed ResetCosmology
* Moved baryon to photon ratio definition to PrimiNuc
* Removed: Incomplete Decoupling of Neutrinos, Neutrino Temperature, Effective Description of Neutrinos, and Scake Factor determination. Can input again just by copying from PRIMIv1.nb.
* Removed comparisons between the Born rate and corrections in Weak Rates section.
* Removed last section: Results and plots as does no calculations.
* Added comments for progress and retimed the nuclear reaction integration.
* Added some new code blocks to PrimiNuc.m.

To Do
-----

* At some point we will want to move the Baryon Density Parameter outside of PrimiNuc
* Get better summary statistics and plots