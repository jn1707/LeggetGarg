import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Read in the mc
neutrinos = np.genfromtxt("./neutrino_mc.csv", delimiter=',', names=True)
muons = np.genfromtxt("./muon_mc.csv", delimiter=',', names=True)

# Read in the hyperplane parameters
hp_nue_cc = np.genfromtxt("./hyperplanes_nue_cc.csv", delimiter=',', names=True)
hp_numu_cc = np.genfromtxt("./hyperplanes_numu_cc.csv", delimiter=',', names=True)
hp_nutau_cc = np.genfromtxt("./hyperplanes_nutau_cc.csv", delimiter=',', names=True)
hp_nu_nc = np.genfromtxt("./hyperplanes_all_nc.csv", delimiter=',', names=True)
hp_muons = np.genfromtxt("hyperplanes_muon.csv", delimiter=',', names=True)

# The bestfit values for Analysis A NC+CC from Table 2
best_fit = {'ice_absorption': 101.5,
            'ice_scattering': 103.0,
            'opt_eff_headon': -0.63,
            'opt_eff_lateral': 0.49,
            'opt_eff_overall': 98.4,
            'coin_fraction': 0.01}


# Write the function definition for neutrinos
# This will take the values for each parameter and return the modified weight for each event
def apply_neutrinos(neutrinos=neutrinos,
                    hyperplanes = {0: hp_nu_nc,
                                   12: hp_nue_cc,
                                   14: hp_numu_cc,
                                   16: hp_nutau_cc,},
                    ice_absorption = 100.0,
                    ice_scattering = 100.0,
                    opt_eff_headon = 0.0,
                    opt_eff_lateral = 0.0,
                    opt_eff_overall = 100.0,
                    coin_fraction = 0.0,
                    **kwargs):
    
    # Copy the weights so we can modify them and 
    # assign an "interaction type" corresponding to
    # nue CC, numu CC, nutau CC or NC
    output_weights = np.copy(neutrinos['weight'])
    int_type = np.copy(neutrinos['pdg']).astype(int)
    int_type[neutrinos['type']==0] = 0 
    int_type = np.abs(int_type)

    # Apply each neutrino flavor separately using the PDG
    # codes for each. Note that we're using 0 for NC interactions
    # and 12/14/16 for nue/numu/nutau CC interactions respectively
    for flavor, hp in hyperplanes.items():
        bins_cz = hyperplanes[flavor]['reco_coszen']
        bins_en = hyperplanes[flavor]['reco_energy']
        bins_pid = hyperplanes[flavor]['pid']
        bins = np.array([bins_cz, bins_en, bins_pid]).T
        
        modifications = hp['offset'] + \
                        hp['ice_scattering'] * (ice_scattering-100)/100. +\
                        hp['ice_absorption'] * (ice_absorption-100)/100. +\
                        hp['opt_eff_lateral'] * (10*opt_eff_lateral) +\
                        hp['opt_eff_headon'] * (opt_eff_headon) +\
                        hp['opt_eff_overall'] * (opt_eff_overall-100)/100. +\
                        hp['coin_fraction'] * (coin_fraction)
        
        # Apply the modifications for a single neutrino flavor
        for i, b in enumerate(bins):
            mask = (int_type==flavor)
            mask &= (neutrinos['reco_coszen'] == b[0])
            mask &= (neutrinos['reco_energy'] == b[1])
            mask &= (neutrinos['pid'] == b[2])
            output_weights[mask] *= modifications[i]
        
    return output_weights

# Write the function definition for muons
# This will take the values for each parameter and return the modified weight for each event
def apply_muons(muons=muons,
                hyperplane = hp_muons,
                ice_absorption = 100.0,
                ice_scattering = 100.0,
                opt_eff_headon = 0.0,
                opt_eff_lateral = 0.0,
                opt_eff_overall = 100.0,
                **kwargs):
    
    # Copy the weights so we can modify them
    output_weights = np.copy(muons['weight'])

    # Get the bins for the muons
    bins_cz = hyperplane['reco_coszen']
    bins_en = hyperplane['reco_energy']
    bins_pid = hyperplane['pid']
    bins = np.array([bins_cz, bins_en, bins_pid]).T
        
    modifications = hyperplane['offset'] +\
        hyperplane['ice_scattering']*(ice_scattering-100)/100. +\
        hyperplane['opt_eff_lateral']*(10*opt_eff_lateral) +\
        hyperplane['opt_eff_headon']*(opt_eff_headon) +\
        hyperplane['ice_absorption']*(np.exp(hyperplane['ice_absorption_expslope']*(ice_absorption/100.-1.0))-1) +\
        hyperplane['opt_eff_overall']*(np.exp(hyperplane['opt_eff_overall_expslope']*(opt_eff_overall/100.-1.0))-1)

    # Apply the modifications to the muon weights
    for i, b in enumerate(bins):
        mask = (muons['reco_coszen'] == b[0])
        mask &= (muons['reco_energy'] == b[1])
        mask &= (muons['pid'] == b[2])
        output_weights[mask] *= modifications[i]
        
    return output_weights


#
# Neutrinos
#

# Make a plot of the neutrinos just to verify the shape is correct
bins_en = np.log10([5.623413,  7.498942, 10. , 
                    13.335215, 17.782795, 23.713737, 
                    31.622776, 42.16965 , 56.23413])
bins_cz = np.array([-1., -0.8, -0.6 , -0.4, -0.2, 
                    0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins_pid = np.array([0, 1, 2])

# Choose the nuisance parameters for the detector systematics
#neutrino_weights = neutrino_muons()
neutrino_weights = apply_neutrinos(**best_fit)

# A flux model must be assumed for the neutrinos
# Here, use a simple flux of phi=800*E^-3.7 
# Note that this is only for the purposes of an
# example: fits performed using this sample should
# use a numerical flux calculation such as the one
# presented in PhysRevD.92.023004
# In principle, this is also where you would apply
# the neutrino oscillation probabilities
neutrino_weights *= 800*neutrinos['true_energy']**-3.7

# Scale to the livetime of the data
neutrino_weights *= 1006*24*3600. 

# Make the histogram, binning in energy, zenith, and pid
nu_hist, edges = np.histogramdd([np.log10(neutrinos['reco_energy']),
                                 neutrinos['reco_coszen'],
                                 neutrinos['pid']],
                                bins = [bins_en, bins_cz, bins_pid],
                                weights = neutrino_weights)

# Make a figure of the cascade neutrinos
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
cmesh = ax1.pcolormesh(bins_en, 
                       bins_cz,
                       nu_hist[:,:,0].T,
                       cmap='Blues')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

cmesh = ax2.pcolormesh(bins_en, 
                       bins_cz,
                       nu_hist[:,:,1].T,
                       cmap='Blues')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

ax1.set_title('Cascade-like, Neutrinos')
ax1.set_ylabel("Cos(Reco Zenith)")
ax1.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_ylabel("Cos(Reco Zenith)")
ax2.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_title('Track-like, Neutrinos')

fig.tight_layout()

plt.savefig('nu_event_dis_cascades_tracks_neutrinos.pdf')


#
# Muons
# 

# Make a plot of the muons just to verify the shape is correct
bins_en = np.log10([5.623413,  7.498942, 10. , 
                    13.335215, 17.782795, 23.713737, 
                    31.622776, 42.16965 , 56.23413])
bins_cz = np.array([-1., -0.8, -0.6 , -0.4, -0.2, 
                    0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins_pid = np.array([0, 1, 2])

# Choose the nuisance parameters for the detector systematics
#muon_weights = np.copy(muons['weight'])
#muon_weights = apply_muons()
muon_weights = apply_muons(**best_fit)

# Weights are in Hz. Convert to about the 
# livetime given in Section IV
muon_weights *= 1006*24*3600. 

# Make the histogram, binning in energy, zenith, and pid
muon_hist, edges = np.histogramdd([np.log10(muons['reco_energy']),
                                   muons['reco_coszen'],
                                   muons['pid']],
                                  bins = [bins_en, bins_cz, bins_pid],
                                  weights = muon_weights)

# Make a figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
cmesh = ax1.pcolormesh(bins_en, 
                       bins_cz,
                       muon_hist[:,:,0].T,
                       cmap='Blues')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

cmesh = ax2.pcolormesh(bins_en, 
                       bins_cz,
                       muon_hist[:,:,1].T,
                       cmap='Blues')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

ax1.set_title('Cascade-like, Atm. Muons')
ax1.set_ylabel("Cos(Reco Zenith)")
ax1.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_ylabel("Cos(Reco Zenith)")
ax2.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_title('Track-like, Atm. Muons')

fig.tight_layout()

plt.savefig('nu_event_dis_cascades_tracks_muons.pdf')


#------------------------------------------------
# End of hyperplanes notebook
#------------------------------------------------


# # Load the data
# neutrino_data = pd.read_csv('neutrino_mc.csv')

# # Print the first few entries of the data
# print(neutrino_data.head())

# # Print all column names
# #print(data.columns)

# # Print all pdg values
# print(neutrino_data['# pdg'].unique())

#------------------------------------------------
# Start Simon's notebook
#------------------------------------------------

# Neutrino oscillation probabilities
def osc_probs_neutrino_mc(neutrinos=neutrinos, 
                       delta_m32: float = 2.53e-3, # eV^2
                       theta_23: float = 49.8, # Degrees
                       **kwargs):
    """
    Calculate neutrino oscillation probabilities.
    For electrons, we set the weights to 1.
    For muons, the return value is P(nu_mu -> nu_mu) = 1 - P(nu_mu -> nu_tau)
    For taus, the return value is P(nu_mu -> nu_tau)
    """
    probabilities = np.ones(neutrinos.shape[0])
    
    is_nu_mu = np.abs(neutrinos['pdg']) == 14
    is_nu_tau = np.abs(neutrinos['pdg']) == 16

    r = 1.3e4/2 # Earth radius [km]
    a = 20 # distance from atmosphere to IceCube [km]
    # Calculate distance traveled by the neutrinos as a function of their true cosZ.
    # This equation can be derived using simple geometry and the law of cosines
    distance = -neutrinos["true_coszen"] * (r-a) + np.sqrt(neutrinos["true_coszen"]**2 * (r-a)**2 - (a**2 - 2*a*r))
    assert np.all(distance >= 20), "Distance is negative!"
    probabilities[is_nu_mu] = 1 - np.sin(2*np.deg2rad(theta_23))**2 * np.sin(1.27 * delta_m32 * distance[is_nu_mu] / neutrinos[is_nu_mu]['true_energy'])**2
    probabilities[is_nu_tau] = np.sin(2*np.deg2rad(theta_23))**2 * np.sin(1.27 * delta_m32 * distance[is_nu_tau] / neutrinos[is_nu_tau]['true_energy'])**2

    assert np.all(probabilities >= 0), "Probabilities are negative!"
    assert np.all(probabilities <= 1), "Probabilities are greater than 1!"
    
    return probabilities

# Interpolate nu_e and nu_mu fluxes from the Honda flux tables
def get_honda_fluxes(filename: str = "nu_flux_honda_table.txt"):
    # Create a dict with the pdg as key and an array as value.
    # The arrays are of shape (n, 3) with the energy, cosZ angle, and flux
    fluxes = {-14: [], -12: [], 12: [], 14: []}

    with open(filename) as f:
        for line in f:
            # If the line starts without a space, it contains the cosZ value range (which we take the average over)
            if line[0] != " ":
                cos1, cos2 = float(line[23:27]), float(line[32:36])
                # To make the interpolation work for all values of cosZ, place the outermost points at the edges, i.e., 1 and -1
                if cos1 == 1. or cos2 == 1.:
                    cosZ = 1.
                elif cos1 == -1. or cos2 == -1.:
                    cosZ = -1
                # For other values of cosZ, take the average of the upper and lower bounds
                else:
                    cosZ = (cos1 + cos2)/2
                continue
            
            # If the line starts with a space but the second , it contains numbers
            elif not line[1].isnumeric():
                continue
            
            # Otherwise, it contains the energy and fluxes
            energy, numu_flux, numubar_flux, nue_flux, nuebar_flux = [float(value) for value in line.split()]
            # Maybe the Honda model uses the opposite convention for cosZ?
            # cosZ = -cosZ
            fluxes[12].append([energy, cosZ, nue_flux])
            fluxes[14].append([energy, cosZ, numu_flux])
            fluxes[-12].append([energy, cosZ, nuebar_flux])
            fluxes[-14].append([energy, cosZ, numubar_flux])
    # TODO check that the angle is read correctly.
    # Convert to numpy array
    fluxes = {key: np.array(value) for key, value in fluxes.items()}
    # Create interpolators for the fluxes
    flux_interpolators = {pdg: scipy.interpolate.LinearNDInterpolator(value[:, :2], value[:, 2]) for pdg, value in fluxes.items()}
    return flux_interpolators

# Outdated function for the expected flux (because it used the muon flux for the tau flux)
def expected_flux_at_detector(neutrinos=neutrinos,
               delta_m32: float = 2.53e-3, # eV^2
               theta_23: float = 49.8, # Degrees
               **kwargs):
    """
    Calculate the flux for the neutrinos based on their true energy and cosZ.
    The Honda flux tables are used.

    Also apply neutrino oscillation probabilities to the weights.
    The weight of a certain (cosZtrue, Etrue) nu_mu is multiplied by P(nu_mu -> nu_mu)
    The weight of a certain (cosZtrue, Etrue) nu_tau is multiplied by P(nu_mu -> nu_tau)
    """
    flux_interpolators = get_honda_fluxes()
    # Apply oscillation probabilities
    oscillation_weights = osc_probs_neutrino_mc(neutrinos=neutrinos,
                                delta_m32=delta_m32,
                                theta_23=theta_23)
    # Multiply oscillation weights with the fluxes, where the nu_tau fluxes are simply the nu_mu fluxes
    # TODO find realistic tau flux
    for pdg in np.unique(neutrinos["pdg"]):
        mask = neutrinos['pdg'] == pdg
        if pdg == 16:
            interpolator = flux_interpolators[14]
        elif pdg == -16:
            interpolator = flux_interpolators[-14]
        else:
            interpolator = flux_interpolators[pdg]

        # Interpolate the flux at the true energy and coszen value
        interpolated_flux = interpolator(neutrinos[mask]['true_energy'], neutrinos[mask]['true_coszen'])
        
        # If the energy of the neutrino is outside the energy range given by the Honda flux table, there are 3 options:
        # - use the flux of the maximum energy allowed by the Honda table, which is 10_000.
        # - drop the events entirely
        # - TODO Use the approximation of E^(-3.7) from the last two points in the Honda table for that zenith angle and extrapolate.
        if np.isnan(interpolated_flux).any():
            interpolated_flux[np.isnan(interpolated_flux)] = 0 #interpolator(1e4, neutrinos[mask][np.isnan(interpolated_flux)]["true_coszen"])
            print (f"Warning: {np.isnan(interpolated_flux).sum()} neutrinos have an energy outside the range of the Honda flux table.")
        
        oscillation_weights[mask] *= interpolated_flux

    # convert from flux in #particles / (m^2 sec sr GeV) to #particles / sec
    # The cross section of DeepCore depends on the zenith angle
    # but I assume it to be the same for all angles for now.
    # In that case, the cross section does not really matter, since we normalize the events to the number of neutrinos.
    # It is approximately 300 m x 300 m x 300 m, based on Fig. 1 in the paper.
    # NOTE: I cannot multiply by the energy, as that results in a massive discrepancy. 
    # I guess the weights already take it into account, or there is something with integration that makes it unnecessary.
    # oscillation_weights *= 300**2 * neutrinos["true_energy"]
    
    return oscillation_weights

# Make a plot of the neutrinos just to verify the shape is correct
bins_en = np.log10([5.623413,  7.498942, 10. , 
                    13.335215, 17.782795, 23.713737, 
                    31.622776, 42.16965 , 56.23413])
bins_cz = np.array([-1., -0.8, -0.6 , -0.4, -0.2, 
                    0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins_pid = np.array([0, 1, 2])

# Choose the nuisance parameters for the detector systematics
#neutrino_weights = neutrino_muons()
neutrino_weights = apply_neutrinos(**best_fit)

# A flux model must be assumed for the neutrinos
# Here, use a simple flux of phi=800*E^-3.7 
# Note that this is only for the purposes of an
# example: fits performed using this sample should
# use a numerical flux calculation such as the one
# presented in PhysRevD.92.023004
# I also apply neutrino oscillation probabilities
neutrino_weights *= expected_flux_at_detector(**best_fit)
# Scale to the livetime of the data (1006 days)
neutrino_weights *= 1006*24*3600. 
# Scale neutrino weights based on the normalization constants for NC and nu_tau nuisance parameters
#neutrino_weights *= neutrino_norm(**best_fit)
# Scale the neutrino weights based on the total number of neutrinos according to table I
#neutrino_weights *= neutrino_total_norm(neutrino_weights=neutrino_weights, n_neutrinos=62203-5022-93)

# Make the histogram, binning in energy, zenith, and pid
nu_hist, edges = np.histogramdd([np.log10(neutrinos['reco_energy']),
                                 neutrinos['reco_coszen'],
                                 neutrinos['pid']],
                                bins = [bins_en, bins_cz, bins_pid],
                                weights = neutrino_weights)
nu_hist = np.swapaxes(nu_hist, 0, 1)

# Calculate the weighted error of each bin on the histogram
# TODO this might be an incorrect way of calculating the statistical error 
# caused by the limited MC statistics
# Source: https://stackoverflow.com/questions/48227037/error-on-weighted-histogram-in-python
nu_err = np.sqrt(
    np.histogramdd([np.log10(neutrinos['reco_energy']),
                    neutrinos['reco_coszen'],
                    neutrinos['pid']],
    bins = [bins_en, bins_cz, bins_pid],
    weights = neutrino_weights**2)[0]
)
# Swap axis 0 and 1 to match the data format and for easier plotting
nu_err = np.swapaxes(nu_err, 0, 1)

# Make a figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
cmesh = ax1.pcolormesh(bins_en, 
                       bins_cz,
                       nu_hist[:,:,0],
                       cmap='Blues')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

cmesh = ax2.pcolormesh(bins_en, 
                       bins_cz,
                       nu_hist[:,:,1],
                       cmap='Blues')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(cmesh, cax)

ax1.set_title('Cascade-like, Neutrinos')
ax1.set_ylabel("Cos(Reco Zenith)")
ax1.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_ylabel("Cos(Reco Zenith)")
ax2.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
ax2.set_title('Track-like, Neutrinos')

fig.tight_layout()

plt.savefig('nu_event_dis_cascades_tracks_neutrinos_Simon.pdf')


# #
# # Plot 
# #


# # Make a plot of the muons just to verify the shape is correct
# bins_en = np.log10([5.623413,  7.498942, 10. , 
#                     13.335215, 17.782795, 23.713737, 
#                     31.622776, 42.16965 , 56.23413])
# bins_cz = np.array([-1., -0.8, -0.6 , -0.4, -0.2, 
#                     0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# bins_pid = np.array([0, 1, 2])

# # Choose the nuisance parameters for the detector systematics
# #muon_weights = np.copy(muons['weight'])
# #muon_weights = apply_muons()
# muon_weights = apply_muons(**best_fit)

# # Weights are in Hz. Convert to about the 
# # livetime given in Section IV
# muon_weights *= 1006*24*3600. 

# # Make the histogram, binning in energy, zenith, and pid
# muon_hist, edges = np.histogramdd([np.log10(muons['reco_energy']),
#                                    muons['reco_coszen'],
#                                    muons['pid']],
#                                   bins = [bins_en, bins_cz, bins_pid],
#                                   weights = muon_weights)

# # Make a figure
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
# cmesh = ax1.pcolormesh(bins_en, 
#                        bins_cz,
#                        muon_hist[:,:,0].T,
#                        cmap='Blues')
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# plt.colorbar(cmesh, cax)

# cmesh = ax2.pcolormesh(bins_en, 
#                        bins_cz,
#                        muon_hist[:,:,1].T,
#                        cmap='Blues')
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# plt.colorbar(cmesh, cax)

# ax1.set_title('Cascade-like, Atm. Muons')
# ax1.set_ylabel("Cos(Reco Zenith)")
# ax1.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
# ax2.set_ylabel("Cos(Reco Zenith)")
# ax2.set_xlabel(r"Log$_{10}$Reco Energy (GeV)")
# ax2.set_title('Track-like, Atm. Muons')

# fig.tight_layout()


# # # Create figure
# # fig, ax = plt.subplots(2, 3, figsize=(10, 7))

# # # Sort the unique pdg values
# # unique_pdg = np.unique(neutrino_data['# pdg'])
# # sorted_pdg = sorted(unique_pdg, key=lambda x: (x < 0, abs(x)))

# # # Plot the number of events as a function of the true energy and true conszen with the same pdg
# # for i, pdg in enumerate(sorted_pdg):
# #     mask = neutrino_data['# pdg'] == pdg
# #     row = i // 3
# #     col = i % 3
# #     im = ax[row, col].hist2d(neutrino_data[mask]['true_energy'], neutrino_data[mask]['true_coszen'], bins=[bins_en, bins_cz])
# #     ax[row, col].set_xlabel('True Energy (GeV)')
# #     ax[row, col].set_ylabel('True Coszen')
# #     ax[row, col].set_title(f'PDG: {pdg}')

# # # Add colorbar
# # fig.colorbar(im[3], ax=ax)

# # # Adjust the layout
# # fig.tight_layout()

# # # Save plot as pdf file with the same name as the python script
# # plt.savefig('/home/20190515_Three-year_high-statistics_neutrino_oscillation_samples/analysis_a/nu_event_dis.pdf')



