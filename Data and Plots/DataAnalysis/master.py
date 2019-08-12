from utils import get_data
from utils import plot_distributions, plot_abundances, plot_chisq_distribution, plot_mchi_omegab_contours 
from utils import plot_joint_mchi_omegab, plot_deltachisq
from utils import get_chisq_grid, get_mass_omegab_grid, get_masses
from utils import chisq
from utils import save_results

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

def run_scenario(scenario):
	filename = scenario + '.txt'
	data = get_data(filename)
	print('Loaded data from: {}\nLength: {}'.format(filename, len(data['mass'])))

	plot_distributions(data, scenario)
	print('[{}] Plotted Distrbutions (1/7)'.format(scenario))
	plot_abundances(data, scenario)
	print('[{}] Plotted Abundances (2/7)'.format(scenario))
	plot_chisq_distribution(data, scenario)
	print('[{}] Plotted Chi Squared Distrbutions (3/7)'.format(scenario))
	plot_mchi_omegab_contours(data, scenario, 'BBN')
	plot_mchi_omegab_contours(data, scenario, 'CMB')
	plot_mchi_omegab_contours(data, scenario, 'BBN+CMB')
	print('[{}] Plotted Omegab vs Mchi Contours (4/7)'.format(scenario))
	plot_joint_mchi_omegab(data, scenario)
	print('[{}] Plotted joint contours (5/7)'.format(scenario))
	plot_deltachisq(data, scenario, zoom=False)
	plot_deltachisq(data, scenario, zoom=True)
	print('[{}] Plotted Delta Chi curves (6/7)'.format(scenario))
	print('[{}] Saving results (7/7)'.format(scenario))
	save_results(data, scenario)

if __name__ == '__main__':
	scenarios = ['EE_Neutral_Scalar', 
			     'EE_Maj', 
			     'Nu_Complex_Scalar', 
			     'Nu_Neutral_Scalar', 
			     'Nu_Zp']
	Parallel(n_jobs=-1)(delayed(run_scenario)(scenario=scenario) for scenario in scenarios)
	

