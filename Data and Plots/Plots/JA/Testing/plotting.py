if __name__ == '__main__':
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np

	neutrino_df = pd.read_csv('EE_results_Dirac.csv', header=0)
	Boehm_Yp_df = pd.read_csv('EEBoehmYp.csv', header=None, names=['mdm', 'Yp'])
	Boehm_DoverH_df = pd.read_csv('EEBoehmDH.csv', header=None, names=['mdm', 'D/H'])
	
	mass_arr = neutrino_df['mDM']
	OmegaB0 = neutrino_df['h2OMegab'][0]
	Yp_arr = neutrino_df['Yp']
	DoverH_arr = neutrino_df['D/H']*10**5


	figsize = (6,6)
	plt.figure(figsize=figsize)

	arrays = [Yp_arr,
			  DoverH_arr]
	mass_arrays = [Boehm_Yp_df['mdm'], Boehm_DoverH_df['mdm']]
	Boehm_arrays = [Boehm_Yp_df['Yp'], Boehm_DoverH_df['D/H']]
	
	column_labels = [r'$Y_p$', 
					 r'$\mathrm{D}/\mathrm{H} \times 10^5$', ]

	colors = ['#419D78',
			  '#FFBF00',]

	markers = ["o", "s"]
	sizes = [60, 60]

	ax1 = plt.subplot(2, 1, 1)
	ax2 = plt.subplot(2, 1, 2)
	axes = [ax1, ax2]

	for idx, color in enumerate(colors):
		axes[idx].plot(mass_arrays[idx], Boehm_arrays[idx],
			c='k',
			ls='-.',
			lw=0.6,
			label='Boehm et al. (2013)')
		# axes[idx].plot(mass_arr, arrays[idx], 
		# 	c='k',
		# 	ls='--',
		# 	lw=0.6,
		# 	label='')
		axes[idx].scatter(mass_arr, arrays[idx], 
			c=colors[idx],
			alpha=0.8,
			s=sizes[idx],
			linewidths=0.4,
			edgecolors='k',
			marker=markers[idx],
			label=column_labels[idx])
		axes[idx].legend()
		axes[idx].set_xlim(10**(-2), 10**(1))
		axes[idx].set_xscale('log')
		axes[idx].set_ylabel(column_labels[idx])
		if idx == 0:
			axes[idx].set_xticklabels([])
		if idx == 1:
			axes[idx].set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$')
	plt.suptitle('Dirac Fermion annihilating into Electrons')
	plt.savefig('EEDiracBoehm.pdf')	

