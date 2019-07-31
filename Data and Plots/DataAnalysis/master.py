from utils import get_data
from utils import plot_distributions, plot_abundances, plot_chisq_distribution, plot_mchi_omegab_contours 
from utils import plot_joint_mchi_omegab, plot_deltachisq
from utils import get_chisq_grid, get_mass_omegab_grid, get_masses
from utils import chisq
from utils import save_results

import warnings
warnings.filterwarnings('ignore')

def run_scenario(scenario):
	filename = scenario + '.txt'
	data = get_data(filename)
	print('Loaded data from: {}\nLength: {}'.format(filename, len(data['mass'])))

	plot_distributions(data, scenario)
	print('Plotted Distrbutions (1/7)')
	plot_abundances(data, scenario)
	print('Plotted Abundances (2/7)')
	plot_chisq_distribution(data, scenario)
	print('Plotted Chi Squared Distrbutions (3/7)')
	plot_mchi_omegab_contours(data, scenario, 'BBN')
	plot_mchi_omegab_contours(data, scenario, 'CMB')
	plot_mchi_omegab_contours(data, scenario, 'BBN+CMB')
	print('Plotted Omegab vs Mchi Contours (4/7)')
	plot_joint_mchi_omegab(data, scenario)
	print('Plotted joint contours (5/7)')
	plot_deltachisq(data, scenario, zoom=False)
	plot_deltachisq(data, scenario, zoom=True)
	print('Plotted Delta Chi curves (6/7)')
	print('Saving results (7/7)')
	save_results(data, scenario)

if __name__ == '__main__':
	scenario = 'Nu_Neutral_Scalar'
	run_scenario(scenario)
	

