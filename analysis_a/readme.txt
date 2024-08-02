Data release of the IceCube/DeepCore 3y high statistics oscillation analysis.
This sample is referred to as 'Analysis A' in [1].
Please refer to [1] for a detailed explanation of the sample and the associated neutrino appearance analyses. 

Note: the instructions contained here are to be applied to Sample A only. This is particularly important
for the detector uncertainties, which have slightly different implementations between the two samples.
For sample B there is a separate, different readme file.


Files:
------
There are seven files containing comma separated values (CSV):
- neutrino_mc.csv : event-by-event information for simulated neutrinos
- muon_mc.csv : the per-bin information from the simulated background muons
- data.csv : the count of actual data events per analysis bin
- hyperplanes_*.csv : the corrections to be applied to the simulated expected count per bin accounting for detector uncertainties


Conventions:
------------
- the cosine of zenith values (`coszen`) are used to encode directional information. The convention
  is as follows: -1 corresponds to straight "upgoing" (= Earth crossing, = from the North) trajectories,
  +1 to straight "downgoing" (= coming directly from sky above, = from the South) trajectories and 0 is horizontal


Binning:
--------
The analysis binning to be used is:
- reco_energy : bin edges = [5.623413,  7.498942, 10. , 13.335215, 17.782795, 23.713737, 31.622776, 42.16965 , 56.23413] in GeV
- reco_coszen : bin edges = [-1., -0.8, -0.6 , -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- pid : 0 = cascades, 1 = tracks

Note that the given values for each event (simulation) or bin (data) are given at the bin centers and not the edges.


neutrino_mc.csv:
----------------
This file contains simulated neutrinos with one event per row. Each event contains the following information:
- true_energy: simulated energy (GeV) of neutrino
- true_coszen: simulated cosine(zenith) of neutrino
- reco_energy: reconstructed energy (GeV) in analysis binning, values are chosen so that events will fall into the correct bins
- reco_coszen: reconstructed cosine(zenith) in analysis binning, values are chosen so that events will fall into the correct bins
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- weight: weight per event, this shall be multiplied by the atmospheric flux in (m^-2 s^-1) and the livetime in (s)
- pdg: PDG code of the neutrino interacting in the detector
- type: interaction type where 0 = any neutral current (NC), 1 = charged current quasi elastic (CC_QE), 2 = charged current resonant (CC_RES), 3 = charged current deep inelastic (CC_DIS), 4 = coherent scattering


muon_mc.csv:
----------
This file contains the simulated atmospheric muon background events with one event per row. Each event contains the following information:
- true_energy: The simulated energy of the muon at generation. The generation occurs on an upright cylinder centered in the detector at IceCube coordinates (x,y,depth) = (0, 0, 2450 m) with a radius of 800 meters and a height of 1600 meters.
- true_coszen: The simulated cosine(zenith) of the muon
- reco_energy: reconstructed energy (GeV) in analysis binning
- reco_coszen: reconstructed cosine(zenith) in analysis binning
- pdg: The PDG code for the simulated muon. All atmospheric muons have a code of 13, with no anti-muons generated for the sample.
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- weight: The weight in units of Hz for the event assuming the H4a cosmic ray spectrum.



data.csv:
--------
This file contains the counts of the observed data:
- reco_energy: reconstructed energy (GeV) in analysis binning
- reco_coszen: reconstructed cosine(zenith) in analysis binning
- pid: reconstructed particle ID, 0 = cascade, 1 = track
- count: count of data events


hyperplanes_*.csv:
------------------
These files contain the information needed to apply the assess the impact from detector uncertainties.
Five different files are following similar structures: four are used for weighting neutrino interactions and one for atmospheric muon interactions.

A usage case and example function is shown in the included jupyter notebook analysisA_detector_systematics

The information from `hyperplanes_all_nc.csv` is applied to any neutral current event (type == 0), `hyperplanes_nue_cc.csv` to the neutrinos that 
have pdg == +12 or -12 (nue and nuebar) and type > 0 (CC), and so forth.

The files contain the usual reco_energy, reco_coszen, pid and 6 additional values:
- offset : constant factor
- ice_absorption : slope accounting for differences in ice optical absorption, nominal parameter values is 1, the uncertainty +/- 10 and the valid range +/- 15
- ice_scattering : slope accounting for differences in ice optical scattering, nominal parameter values is 1, the uncertainty +/- 10 and the valid range +/- 15
- opt_eff_overall : slope accounting for overall optical efficiency of DOMs, nominal parameter value is 1.0, the uncertainty +/- 0.1 and the valid range is [0.8, 1.2]
- opt_eff_lateral : slope accounting for lateral optical efficiency of DOMs, nominal parameter value is 25, the uncertainty +/- 10 and the valid range is [5, 50]
- opt_eff_headon : slope accounting for head-on optical efficiency of DOMs, nominal parameter value is 0, the uncertainty unknown (flat) and the valid range is [-5.0, +2.0]

The neutrino files contain an additional column as well: 
- coin_fraction : slope describing the uncertainty in the neutrino histogram shapes due to random coincidences between neutrinos and atmospheric muons from unrelated showers. The nominal parameter value is 0 and the uncertainty is treated as a one-sided gaussian prior with a width of 0.1. The parameter range is [0, 1.0].

A multiplicative reweighting factor for each analysis bin is produced given a set of values for the detector systematic uncertainties. The reweighting factor, `f`, is calculated from the information in the hyperplanes_*.csv file ('df') in this way:

`f = (df['offset'] 
      + df['ice_scattering']*(value-100)/100
      + df['opt_eff_lateral']*(10*value)
      + df['opt_eff_headon']*(value)
      + df['ice_absorption']*(value-100.)/100
      + df['opt_eff_overall']*(value-100)/100.
      + df['coin_fraction']*(value) )`

where "value" is in the form given in Table II. The factor `f` is then applied multiplicatively to the final event rates for the given bin.

The atmospheric muons have additional complications due to a different functional form. The files contain the usual reco_energy, reco_coszen, pid and 4 values identical to those described for the neutrinos:
- offset : constant factor
- ice_scattering : slope accounting for differences in ice optical scattering, nominal parameter values is 1, the uncertainty +/- 10 and the valid range +/- 15
- opt_eff_lateral : slope accounting for lateral optical efficiency of DOMs, nominal parameter value is 25, the uncertainty +/- 10 and the valid range is [5, 50]
- opt_eff_headon : slope accounting for head-on optical efficiency of DOMs, nominal parameter value is 0, the uncertainty unknown (flat) and the valid range is [-5.0, +2.0]

The parametrization for the optical absorption in the ice and the overall optical efficiency are parametrized using an exponential form for the muons, leading to a difference in meaning of the parameters.

- ice_absorption : normalization constant accounting for differences in ice optical absorption, nominal parameter values is 1 and a flat prior is used.
- ice_absorption_expslope : exponential slope describing the impact of ice optical absorption
- opt_eff_overall : normalization constant accounting for uncertainties in the overall optical efficiency of DOMs, nominal parameter value is 1.0, the uncertainty +/- 0.1 and the valid range is [0.8, 1.2]
- opt_eff_overall_expslope: exponential slop describing the impact of the overall optical efficiency of DOMs

The reweighting factor for each analysis bin is calculated in this way:
`f = (df['offset'] 
      + df['ice_scattering']*((value-100)/100.)
      + df['opt_eff_lateral']*(10*value)
      + df['opt_eff_headon']*(value)
      + df['ice_absorption'] * (exp(df['ice_absorption_expslope']*(value/100.-1.0)) - 1.0)
      + df['opt_eff_overall'] * (exp(df['opt_eff_overall_expslope']*(value/100.-1.0)) - 1.0) )`

This factor `f` may then be used as a multiplicative modification of the rate in the given bin.


References:
-----------
[1] M. G. Aartsen et al. Measurement of Atmospheric Tau Neutrino Appearance with IceCube DeepCore. Phys. Rev., D99(3):032007, 2019. arXiv:1901.05366, doi:10.1103/PhysRevD.99.032007
[2] M. G. Aartsen et al. Measurement of Atmospheric Neutrino Oscillations at 6–56 GeV with IceCube DeepCore. Phys. Rev. Lett., 120(7):071801, 2018. arXiv:1707.07081, doi:10.1103/PhysRevLett.120.071801

