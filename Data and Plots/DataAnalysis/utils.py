import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def get_data(filename):
    data = np.loadtxt(filename, skiprows=1)
    mass = data[:, 0]
    OmegaB = data[:, 1]
    Neff = data[:, 2]
    H = data[:, 4]
    D = data[:, 5]
    He = data[:, 8]
    DoverH = D / H
    Yp = 4 * He
    return {'mass': mass, 
            'OmegaB': OmegaB, 
            'Neff': Neff, 
            'H': H, 
            'D': D, 
            'He': He, 
            'D/H': DoverH, 
            'Yp': Yp}

def plot_distributions(data, scenario):
    plt.figure(figsize=(10, 5))
    bins = 40
    alpha = 0.6
    ax = plt.subplot(1, 2, 1)
    ax.hist(data['Yp'], 
            bins=bins, 
            density=True, 
            alpha=alpha)
    ax.set_xlim(np.min(data['Yp']), np.max(data['Yp']))
    ax.set_xlabel(r'$Y_p$')
    ax.set_ylabel(r'$\mathrm{Density}$')
    ax = plt.subplot(1, 2, 2)
    ax.hist(data['D/H'] * 10**5, 
            bins=bins, 
            density=True, 
            alpha=alpha)
    ax.set_xlabel(r'$\mathrm{D}/\mathrm{H} \times 10^5$')
    ax.set_xlim(np.min(data['D/H'])*10**5, np.max(data['D/H'])*10**5)
    plt.suptitle('Distributions')
    plt.savefig(scenario + '_abundance_distributions.pdf')

def plot_abundances(data, scenario):
    YpCentre = 0.245
    YpError = 0.003
    DHCentre = 2.569 * 10**(-5)
    DHError = 0.027 * 10**(-5)
    
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1,100)
    ax1 = fig.add_subplot(gs[0, 0:40])
    ax2 = fig.add_subplot(gs[0, 54:100])
    ax1.scatter(data['mass'], data['Yp'],
               lw=0.0,
               marker='.',
               s=20,
               alpha=0.8,
               c=data['OmegaB'],
               cmap='PiYG')
    ax1.set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$')
    ax1.set_ylabel(r'$Y_p$')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, np.max(data['mass']))
    ax1.add_patch(Rectangle(xy=(0.1, YpCentre - YpError),
                            width=(30.0 - 0.1),
                            height=2*YpError,
                            alpha=0.1,
                            color='k'))
    ax2.add_patch(Rectangle(xy=(0.1, (DHCentre - DHError)*10**5),
                            width=(30.0 - 0.1),
                            height=2*DHError*10**5,
                            alpha=0.1,
                            color='k'))
    sc = ax2.scatter(data['mass'], data['D/H'] * 10**5, 
               lw=0.0,
               marker='.',
               s=20,
               alpha=0.8,
               c=data['OmegaB'],
               cmap='PiYG')
    ax2.set_xlabel(r'$m_{\chi} \, \mathrm{[MeV]}$')
    ax2.set_ylabel(r'$\mathrm{D}/\mathrm{H} \times 10^5$')
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, np.max(data['mass']))
    plt.colorbar(sc, ax=ax2, label=r'$\Omega_b h^2$')
    plt.suptitle('Dependence on Dark Matter Mass')
    plt.savefig(scenario + '_abundances.pdf')

def plot_chisq_distribution(data, scenario):
    plt.figure(figsize=(5,5))
    bins = 40
    alpha = 1.0
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='BBN'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='BBN')
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='CMB'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='CMB')
    plt.hist(chisq(data['Yp'], data['D/H'], data['OmegaB'], data['Neff'], type='BBN+CMB'),
             bins=bins, 
             density=True,
             histtype='step',
             alpha=alpha,
             lw=1.7,
             label='BBN+CMB')
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('Density')
    plt.legend()
    plt.yscale('log')
    plt.title('Distribution of $\chi^2$ Values', fontsize=16)
    plt.savefig(scenario + '_chisq_distributions.pdf')

def plot_mchi_omegab_contours(data, scenario, type):
    mass_grid, omegab_grid = get_mass_omegab_grid(data)
    chisq_grid = get_chisq_grid(data, type=type)
    
    confidence_levels = [2.30, 6.18, 11.83, 19.33, 28.74]
    plt.figure(figsize=(5, 5))
    plt.suptitle(type)
    ax = plt.subplot(1,1,1)
    ct = ax.contour(mass_grid, omegab_grid, chisq_grid,
                    levels=confidence_levels,
                    cmap='PiYG')
    ct.levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    ax.set_ylabel(r'$\Omega_b h^2$')
    ax.set_xlim(0.1,)
    ax.set_ylim(0.020, 0.025)
    ax.clabel(ct, ct.levels, inline=True, fontsize=16)
    if type == 'BBN':
        type_str = 'bbn'
    elif type == 'CMB':
        type_str = 'cmb'
    elif type == 'BBN+CMB':
        type_str = 'bbnandcmb'
    plt.savefig(scenario + '_' + type_str + '_mchiomegab.pdf')

def plot_joint_mchi_omegab(data, scenario):
    confidence_levels = [0.0, 6.18, 28.74]

    MASS, OMEGAB = get_mass_omegab_grid(data)
    CHISQBBN = get_chisq_grid(data, type='BBN')
    CHISQBBNandCMB = get_chisq_grid(data, type='BBN+CMB')

    plt.figure(figsize=(7,6))
    plt.suptitle('Combining Measurements')
    ax = plt.subplot(1,1,1)
    ax.set_xlim(0.1,11)
    ax.set_ylim(0.020, 0.028)
    ct = ax.contourf(MASS, OMEGAB, CHISQBBN, 
                     levels=confidence_levels,
                     colors=['#8B9DB6','#01295F',],
                     alpha=0.3,)
    ct = ax.contourf(MASS, OMEGAB, CHISQBBNandCMB, 
                     levels=confidence_levels,
                     colors=['#419D78', '#368163'],
                     alpha=0.8)
    proxy = [plt.Rectangle((0,0),1,1,fc='#01295F',alpha=0.3), plt.Rectangle((0,0),1,1,fc='#419D78',alpha=0.7)]
    ax.set_xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    ax.set_ylabel(r'$\Omega_b h^2$')
    ax.legend(proxy, [r"BBN", r"BBN+CMB"], fontsize=14)
    plt.savefig(scenario + '_exclusion.pdf')

def plot_deltachisq(data, scenario, zoom=False):
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB')
    masses = get_masses(data)

    MC, CSQ = np.meshgrid(np.logspace(-1, 2), np.linspace(0, 40))
    col = 'k'

    plt.figure(figsize=(8,6))
    plt.plot(masses, bbninterpfn(masses),lw=1.7,label='BBN',c=col,ls='-.')
    plt.plot(masses, cmbinterpfn(masses),lw=1.7,label='CMB',c=col,ls=':')
    plt.plot(masses, bbnandcmbinterpfn(masses),lw=1.7,label='BBN+CMB',c=col,ls='-')
    confidence_levels = [1.0, 4.0, 9.0, 16.0, 25.0]
    ct = plt.gca().contour(MC, CSQ, CSQ, 
                    levels=confidence_levels,
                    cmap='PiYG')
    ct.levels = [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$', r'$5\sigma$']
    plt.xlabel(r'$m_\chi \, \mathrm{[MeV]}$')
    plt.ylabel(r'$\Delta \tilde{\chi}^2_{\mathrm{BBN}}$')
    plt.xscale('log')
    plt.gca().set_xlim(0.1,15)
    if zoom:
        plt.gca().set_ylim(0, 40)
    else:
        plt.gca().set_ylim(0, 80)
    plt.gca().clabel(ct, ct.levels, inline=True, fontsize=24)
    plt.legend(fontsize=14, frameon=True)
    plt.tick_params(axis='x', which='minor', size=4)
    if zoom:
        plt.savefig(scenario + '_chisqmarg_zoom.pdf')
    else:
        plt.savefig(scenario + '_chisqmarg_full.pdf')

def get_mass_omegab_grid(data):
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    return MASS, OMEGAB

def get_chisq_grid(data, type):
    masses = np.unique(data['mass'])
    omegabs = np.unique(data['OmegaB'])
    MASS, OMEGAB = np.meshgrid(masses, omegabs)
    OMEGABDAT = data['OmegaB'].reshape(len(masses), -1).T
    YP = data['Yp'].reshape(len(masses), -1).T
    DH = data['D/H'].reshape(len(masses), -1).T
    NEFF = data['Neff'].reshape(len(masses), -1).T
    return chisq(YP, DH, OMEGABDAT, NEFF, type)

def get_masses(data):
    return np.unique(data['mass'])

def chisqBBN(Yp, DoverH):
    YpCentre = 0.245
    YpError = 0.003
    YpErrorTh = 0.00017
    DHCentre = 2.569 * 10**(-5)
    DHError = (0.027) * 10**(-5)
    DHErrorTh = 0.036 * 10**(-5)
    
    return np.power(Yp - YpCentre, 2)/(YpError**2 + YpErrorTh**2) \
         + np.power(DoverH - DHCentre, 2)/(DHError**2 + DHErrorTh**2)

def chisqCMB(OmegaB, Neff, Yp):
    OmegaBCentre = 0.02225
    NeffCentre = 2.89
    YpCentre = 0.246
    dO = OmegaB - OmegaBCentre
    dN = Neff - NeffCentre
    dY = Yp - YpCentre
    rho12 = 0.40
    rho13 = 0.18
    rho23 = -0.69
    Delta1 = 0.00022
    Delta2 = 0.31
    Delta3 = 0.018
    # rho12 = 0.8294866703898539
    # rho13 = 0.2931412125354891
    # rho23 = -0.19678653064061352
    # Delta1 = 0.00016615671572250368
    # Delta2 = 0.09439790664795518
    # Delta3 = 0.006823614334675781
    SigmaCMB = np.array([[Delta1**2, Delta1*Delta2*rho12, Delta1*Delta3*rho13], 
                         [Delta1*Delta2*rho12, Delta2**2, Delta2*Delta3*rho23], 
                         [Delta1*Delta3*rho13, Delta2*Delta3*rho23, Delta3**2]])
    inv = np.linalg.inv(SigmaCMB)
    
    return dO*dO*inv[0][0] + dN*dN*inv[1][1] + dY*dY*inv[2][2] + 2*dO*dN*inv[0][1] + 2*dO*dY*inv[0][2] + 2*dN*dY*inv[1][2]

def chisqBBNandCMB(Yp, DoverH, OmegaB, Neff):
    return chisqBBN(Yp, DoverH) + chisqCMB(OmegaB, Neff, Yp)

def chisq(Yp, DoverH, OmegaB, Neff, type):
    if type == 'BBN':
        return chisqBBN(Yp, DoverH)
    elif type == 'CMB':
        return chisqCMB(OmegaB, Neff, Yp)
    elif type == 'BBN+CMB':
        return chisqBBNandCMB(Yp, DoverH, OmegaB, Neff)


def get_chisq_marg_interpolation(data, type, kind='cubic'):
    CHISQ = get_chisq_grid(data, type)
    masses = get_masses(data)
    chisq_marg = np.empty(len(masses))
    for idx, mass in enumerate(masses):
        chisq_marg[idx] = np.min(CHISQ[:, idx])
    chisq_marg_rescaled = chisq_marg - np.min(chisq_marg)
    return interp1d(masses, chisq_marg_rescaled, kind=kind)

def save_results(data, scenario, x0=1.0, save=True):
    bbninterpfn = get_chisq_marg_interpolation(data, type='BBN', kind='cubic')
    cmbinterpfn = get_chisq_marg_interpolation(data, type='CMB', kind='cubic')
    bbnandcmbinterpfn = get_chisq_marg_interpolation(data, type='BBN+CMB', kind='cubic')
    
    bbnbounds = []
    cmbbounds = []
    bbnandcmbbounds = []
    for level in [1.0, 4.0, 9.0, 16.0, 25.0]:
        tempbbnbounds = []
        tempcmbbounds = []
        tempbbnandcmbbounds = []
        def bbntempfn(mchi):
            return bbninterpfn(mchi) - level
        def cmbtempfn(mchi):
            return cmbinterpfn(mchi) - level
        def bbnandcmbtempfn(mchi):
            return bbnandcmbinterpfn(mchi) - level
        try:      
            tempbbnbounds.append(fsolve(bbntempfn, x0=0.01)[0])
        except:
            tempbbnbounds.append(0.0)
        try:      
            tempbbnbounds.append(fsolve(bbntempfn, x0=0.1)[0])
        except:
            tempbbnbounds.append(0.0)
        try:      
            tempbbnbounds.append(fsolve(bbntempfn, x0=1.0)[0])
        except:
            tempbbnbounds.append(0.0)
        try:      
            tempbbnbounds.append(fsolve(bbntempfn, x0=x0)[0])
        except:
            tempbbnbounds.append(0.0)
        bbnbounds.append(np.max(tempbbnbounds))
        try:      
            tempcmbbounds.append(fsolve(cmbtempfn, x0=0.01)[0])
        except:
            tempcmbbounds.append(0.0)
        try:      
            tempcmbbounds.append(fsolve(cmbtempfn, x0=0.1)[0])
        except:
            tempcmbbounds.append(0.0)
        try:      
            tempcmbbounds.append(fsolve(cmbntempfn, x0=1.0)[0])
        except:
            tempcmbbounds.append(0.0)
        try:      
            tempcmbbounds.append(fsolve(cmbntempfn, x0=10.0)[0])
        except:
            tempcmbbounds.append(0.0)
        try:      
            tempcmbbounds.append(fsolve(cmbntempfn, x0=x0)[0])
        except:
            tempcmbbounds.append(0.0)
        cmbbounds.append(np.max(tempcmbbounds))
        try:      
            tempbbnandcmbbounds.append(fsolve(bbnandcmbtempfn, x0=0.01)[0])
        except:
            tempbbnandcmbbounds.append(0.0)
        try:      
            tempbbnandcmbbounds.append(fsolve(bbnandcmbtempfn, x0=0.1)[0])
        except:
            tempbbnandcmbbounds.append(0.0)
        try:      
            tempbbnandcmbbounds.append(fsolve(bbnandcmbntempfn, x0=1.0)[0])
        except:
            tempbbnandcmbbounds.append(0.0)
        try:      
            tempbbnandcmbbounds.append(fsolve(bbnandcmbntempfn, x0=x0)[0])
        except:
            tempbbnandcmbbounds.append(0.0)
        bbnandcmbbounds.append(np.max(tempbbnandcmbbounds))

    if save:
        with open(scenario + '_results.txt', 'w') as f:
            print("-----------------------------------------------------------", file=f)
            print(" Conf. Lev. \t BBN \t\t CMB \t\t BBN + CMB", file=f)
            print("-----------------------------------------------------------", file=f)
            for idx,_ in enumerate(bbnbounds):
                print(" {} sigma \t {:.2f} MeV \t {:.2f} MeV \t {:.2f} MeV".format(idx + 1, bbnbounds[idx], cmbbounds[idx], bbnandcmbbounds[idx]), file=f)
            print("-----------------------------------------------------------", file=f)
            print("")
            print("NOTE: A value of 0.0 MeV means no bound could be determined.", file=f)
    else:
        print("-----------------------------------------------------------")
        print(" Conf. Lev. \t BBN \t\t CMB \t\t BBN + CMB")
        print("-----------------------------------------------------------")
        for idx,_ in enumerate(bbnbounds):
            print(" {} sigma \t {:.2f} MeV \t {:.2f} MeV \t {:.2f} MeV".format(idx + 1, bbnbounds[idx], cmbbounds[idx], bbnandcmbbounds[idx]))
        print("-----------------------------------------------------------")
        print("")
        print("NOTE: A value of 0.0 MeV means no bound could be determined.")
