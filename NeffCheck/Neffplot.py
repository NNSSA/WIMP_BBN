if __name__ == '__main__':
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np

	primat_df = pd.read_csv('neffprimat.csv', header=0)
	primi_df = pd.read_csv('neffprimi.csv', header=0)
	for df in [primat_df, primi_df]:
		df['D/H x 10^5'] = df['D/H'] * 10**5
		df['3He/H x 10^5'] = df['3He/H'] * 10**5
		df['7Li/H x 10^11'] = df['7Li/H'] * 10**11

	columns = ['H', 
			   'Yp', 
			   'D/H x 10^5', 
			   '3He/H x 10^5', 
			   '7Li/H x 10^11']
	
	column_labels = [r'$\mathrm{H}$', 
					 r'$Y_p$', 
					 r'$D/H \times 10^5$', 
					 r'$^{3}\mathrm{He}/\mathrm{H} \times 10^{5}$', 
					 r'$^{7}\mathrm{Li}/\mathrm{H} \times 10^{11}$']

	figsize = (18, 18)
	plt.figure(figsize=figsize)
	primat_color = '#01295F'
	primi_color = '#D1495B'
	diff_color = '#419D78'

	for row, column in enumerate(columns):
		ax = plt.subplot(5, 3, 3*row + 1)
		if row == 4:
			ax.set_xlabel(r'$\Delta N_{\mathrm{eff}}$')
		ax.set_ylabel(column_labels[row])
		ax.plot(primat_df['DeltaNeff'], primat_df[column], c=primat_color, lw=1.7)
		if row == 0:
			ax.set_title('PRIMAT')

		ax = plt.subplot(5, 3, 3*row + 2)
		if row == 4:
			ax.set_xlabel(r'$\Delta N_{\mathrm{eff}}$')
		ax.plot(primi_df['DeltaNeff'], primi_df[column], c=primi_color, lw=1.7)
		if row == 0:
			ax.set_title('PRIMI')

		ax = plt.subplot(5, 3, 3*row + 3)
		if row == 4:
			ax.set_xlabel(r'$\Delta N_{\mathrm{eff}}$')
		ax.set_ylabel('Rel. Diff.')
		ax.semilogy(primi_df['DeltaNeff'], np.abs(1 - (primi_df[column]/primat_df[column])), c=diff_color, lw=1.7)
		if row == 0:
			ax.set_title('Difference')

	plt.savefig('neffcheck.pdf')