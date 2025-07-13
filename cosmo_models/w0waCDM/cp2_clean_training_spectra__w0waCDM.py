import sys
import os
#os.environ['OMP_NUM_THREADS'] = '2'
#os.environ['MKL_NUM_THREADS'] = '2'
#os.environ['OPENBLAS_NUM_THREADS'] = '2'
#os.environ['NUMEXPR_NUM_THREADS'] = '2'
#os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
import numpy as np
from scipy.constants import c
import classy

filepath_params = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM/LHC2e5__LCDM__parameter_file.npz'
batch = int(sys.argv[2])
try:
    input_i = int(sys.argv[3])
except:
    input_i = -1
batch_size = 1000

path, filename_params = filepath_params.rsplit('/', 1)
cube = filename_params.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_params.split('__')[1]
assert model == 'w0waCDM'
n_samples = cube[3:]
assert n_samples in ['1e3', '1e4', '2e5']
n_samples = int(float(n_samples))
assert n_samples == 1000 or n_samples == 10000 or n_samples == 200000
assert path == f"{cube}__{model}"
assert filepath_params == f"{path}/{cube}__{model}__parameter_file.npz"

params_LHC = np.load(filepath_params)
params_LHC_keys = np.array(list(params_LHC.keys()))
n_params = params_LHC_keys.size
assert params_LHC['ID'].size == n_samples
print(f"{filepath_params}, model={model}, n_samples={n_samples}, batch={batch}", flush=True)

k_modes = np.loadtxt(f"{path}/{cube}__{model}__k_modes.txt")
z_modes = np.loadtxt(f"{path}/{cube}__{model}__z_modes.txt")
l_modes = np.loadtxt(f"{path}/{cube}__{model}__l_modes.txt", dtype=int)
kmax = k_modes[-1]
zmax = z_modes[-1]
lmax = l_modes[-1]
params_derived = list(np.loadtxt(f"{path}/{cube}__{model}__derived_params.txt", dtype=str))

observables = ['TT', 'TE', 'EE', 'PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z', 'DER']
header = {
    'TT': f"{params_LHC_keys}, TT(ell)",
    'TE': f"{params_LHC_keys}, TE(ell)",
    'EE': f"{params_LHC_keys}, EE(ell)",
    'PP': f"{params_LHC_keys}, PP(ell)",
    'PKL': f"{params_LHC_keys}, P_lin(k)",
    'PKNL': f"{params_LHC_keys}, P_nonlin(k)",
    'DAZ': f"{params_LHC_keys}, d_angular(z)",
    'HZ': f"{params_LHC_keys}, H(z)",
    'S8Z': f"{params_LHC_keys}, sigma8(z)",
    'DER': f"{params_LHC_keys}, {params_derived}"
}
TT_array   = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__TT_batch{batch:03d}.dat"))
TE_array   = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__TE_batch{batch:03d}.dat"))
EE_array   = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__EE_batch{batch:03d}.dat"))
PP_array   = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__PP_batch{batch:03d}.dat"))
PKL_array  = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__PKL_batch{batch:03d}.dat"))
PKNL_array = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__PKNL_batch{batch:03d}.dat"))
DAZ_array  = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__DAZ_batch{batch:03d}.dat"))
HZ_array   = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__HZ_batch{batch:03d}.dat"))
S8Z_array  = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__S8Z_batch{batch:03d}.dat"))
DER_array  = np.atleast_2d(np.loadtxt(f"{path}/{cube}__{model}__DER_batch{batch:03d}.dat"))
assert TT_array[-1, 0]   >= batch * batch_size + 990, f"{TT_array[-1, 0]}"
assert TE_array[-1, 0]   >= batch * batch_size + 990, f"{TE_array[-1, 0]}"
assert EE_array[-1, 0]   >= batch * batch_size + 990, f"{EE_array[-1, 0]}"
assert PP_array[-1, 0]   >= batch * batch_size + 990, f"{PP_array[-1, 0]}"
assert PKL_array[-1, 0]  >= batch * batch_size + 990, f"{PKL_array[-1, 0]}"
assert PKNL_array[-1, 0] >= batch * batch_size + 990, f"{PKNL_array[-1, 0]}"
assert DAZ_array[-1, 0]  >= batch * batch_size + 990, f"{DAZ_array[-1, 0]}"
assert HZ_array[-1, 0]   >= batch * batch_size + 990, f"{HZ_array[-1, 0]}"
assert S8Z_array[-1, 0]  >= batch * batch_size + 990, f"{S8Z_array[-1, 0]}"
assert DER_array[-1, 0]  >= batch * batch_size + 990, f"{DER_array[-1, 0]}"
# make a backup
np.save(f"{path}/{cube}__{model}__TT_batch{batch:03d}_",   TT_array,   allow_pickle=False)
np.save(f"{path}/{cube}__{model}__TE_batch{batch:03d}_",   TE_array,   allow_pickle=False)
np.save(f"{path}/{cube}__{model}__EE_batch{batch:03d}_",   EE_array,   allow_pickle=False)
np.save(f"{path}/{cube}__{model}__PP_batch{batch:03d}_",   PP_array,   allow_pickle=False)
np.save(f"{path}/{cube}__{model}__PKL_batch{batch:03d}_",  PKL_array,  allow_pickle=False)
np.save(f"{path}/{cube}__{model}__PKNL_batch{batch:03d}_", PKNL_array, allow_pickle=False)
np.save(f"{path}/{cube}__{model}__DAZ_batch{batch:03d}_",  DAZ_array,  allow_pickle=False)
np.save(f"{path}/{cube}__{model}__HZ_batch{batch:03d}_",   HZ_array,   allow_pickle=False)
np.save(f"{path}/{cube}__{model}__S8Z_batch{batch:03d}_",  S8Z_array,  allow_pickle=False)
np.save(f"{path}/{cube}__{model}__DER_batch{batch:03d}_",  DER_array,  allow_pickle=False)

for i in range(batch_size):
    if input_i >= 0:
        i = input_i
    idx = batch*batch_size+i
    if idx in TT_array[:, 0] and idx in TE_array[:, 0] and idx in EE_array[:, 0] and idx in PP_array[:, 0] and idx in PKL_array[:, 0] and idx in PKNL_array[:, 0] and idx in DAZ_array[:, 0] and idx in HZ_array[:, 0] and idx in S8Z_array[:, 0] and idx in DER_array[:, 0]:
        print(f"{idx:06d}done", end=' ', flush=True)
        if input_i >= 0:
            break
        else:
            continue
    else:
        print(flush=True)
        print(batch, i, idx, " TT:", TT_array[i-2:i+2, 0], " PKL:", PKL_array[i-2:i+2, 0], " DAZ:", DAZ_array[i-2:i+2, 0], " DER:", DER_array[i-2:i+2, 0])
    # Define your cosmology (what is not specified will be set to CLASS default parameters)
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'non_linear':'hmcode',
        'l_max_scalars': lmax,
        'z_max_pk': zmax,
        'P_k_max_1/Mpc': kmax,
        'nonlinear_min_k_max': 20.0,
        'accurate_lensing': 1,
        'k_max_tau0_over_l_max': 3.0,
        'perturbations_sampling_stepsize': 0.05,
        'a_ini_over_a_today_default': 1.e-16,
        'N_ncdm': 1,
        'N_ur': 2.0328,
        'm_ncdm': 0.06,
        'Omega_Lambda': 0,
        'use_ppf': 'yes',
        'omega_b':     params_LHC['omega_b'][idx],
        'omega_cdm':   params_LHC['omega_cdm'][idx],
        'H0':          params_LHC['H0'][idx],
        'tau_reio':    params_LHC['tau_reio'][idx],
        'ln_A_s_1e10': params_LHC['logA'][idx],
        'n_s':         params_LHC['n_s'][idx],
        'w0_fld':      params_LHC['w'][idx],
        'wa_fld':      params_LHC['wa'][idx],
    }
    if params['w0_fld'] == -1 and params['wa_fld'] == 0:
        params.pop('Omega_Lambda')
        params.pop('use_ppf')
        params.pop('w0_fld')
        params.pop('wa_fld')
    z_idx = params_LHC['z'][idx]

    try:
        succeeded = False
        while not succeeded:
            try:
                # Set the parameters to the cosmological code
                cosmo = classy.Class()
                cosmo.set(params)
                cosmo.compute()
                succeeded = True
            except classy.CosmoComputationError as e:
                print("succeeded =", succeeded, flush=True)
                cosmo.struct_cleanup()
                cosmo.empty()
                if "tau_sup < pth->tau_reio" in e.message and 'reionization_z_start_max' not in params:
                    print("Changing 'reionization_z_start_max' to 50.", end=' ')
#class_precision_parameter(reionization_z_start_max,double,50.0) /**< Maximum starting value in z for reionization */
                    params['reionization_z_start_max'] = 50.
                elif "tau_sup < pth->tau_reio" in e.message and params['reionization_z_start_max'] < 60.:
                    print("Changing 'reionization_z_start_max' to 60.", end=' ')
                    params['reionization_z_start_max'] = 60.
                elif "tau_sup < pth->tau_reio" in e.message and params['reionization_z_start_max'] < 75.:
                    print("Changing 'reionization_z_start_max' to 75.", end=' ')
                    params['reionization_z_start_max'] = 75.
                elif "tau_sup < pth->tau_reio" in e.message and params['reionization_z_start_max'] < 100.:
                    print("Changing 'reionization_z_start_max' to 100.", end=' ')
                    params['reionization_z_start_max'] = 100.
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and 'background_Nloga' not in params and 'tol_background_integration' not in params:
                    print("Setting 'background_Nloga' to 100000 and 'tol_background_integration' to 1e-14 and 'perturbations_sampling_stepsize' to 0.02.", end=' ')
#class_precision_parameter(background_Nloga,int,40000)
#class_precision_parameter(tol_background_integration,double,1.e-10)
                    params['background_Nloga'] = 100000
                    params['tol_background_integration'] = 1.e-14
                    params['perturbations_sampling_stepsize'] = 0.02
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and 'tol_perturbations_integration' not in params:
                    print("Changing 'tol_perturbations_integration' to 1.e-6", end=' ')
                    params['tol_perturbations_integration'] = 1.e-6
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 1.e-5:
                    print("Changing 'tol_perturbations_integration' to 1.e-5", end=' ')
                    params['tol_perturbations_integration'] = 1.e-5
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 1.e-4:
                    print("Changing 'tol_perturbations_integration' to 1.e-4", end=' ')
                    params['tol_perturbations_integration'] = 1.e-4
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 2.e-4:
                    print("Changing 'tol_perturbations_integration' to 2.e-4", end=' ')
                    params['tol_perturbations_integration'] = 2.e-4
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 3.e-4:
                    print("Changing 'tol_perturbations_integration' to 3.e-4", end=' ')
                    params['tol_perturbations_integration'] = 3.e-4
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 5.e-4:
                    print("Changing 'tol_perturbations_integration' to 5.e-4", end=' ')
                    params['tol_perturbations_integration'] = 5.e-4
                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and params['tol_perturbations_integration'] < 1.e-3:
                    print("Changing 'tol_perturbations_integration' to 1.e-3", end=' ')
                    params['tol_perturbations_integration'] = 1.e-3
                elif "absh <= hmin" in e.message and params['a_ini_over_a_today_default'] > 1.e-17:
                    print("Changing 'a_ini_over_a_today_default' to 1.e-17", end=' ')
                    params['a_ini_over_a_today_default'] = 1.e-17
                elif "absh <= hmin" in e.message and params['a_ini_over_a_today_default'] > 1.e-22:
                    print("Changing 'a_ini_over_a_today_default' to 1.e-22", end=' ')
                    params['a_ini_over_a_today_default'] = 1.e-22
                elif "absh <= hmin" in e.message and params['a_ini_over_a_today_default'] > 1.e-25:
                    print("Changing 'a_ini_over_a_today_default' to 1.e-25", end=' ')
                    params['a_ini_over_a_today_default'] = 1.e-25
                elif "error in background_solve" in e.message and "absh <= hmin" in e.message and params['tol_background_integration'] < 1.e-11:
                    print("Changing 'tol_background_integration' to 1.e-11", end=' ')
                    params['tol_background_integration'] = 1.e-11
                elif "error in background_solve" in e.message and "absh <= hmin" in e.message and params['tol_background_integration'] < 1.e-10:
                    print("Changing 'tol_background_integration' to 1.e-10", end=' ')
                    params['tol_background_integration'] = 1.e-10
#                elif "error in background_solve" in e.message and "absh <= hmin" in e.message and 'background_evolver' not in params:
#                    print("Changing 'background_evolver' to 0 for 'rk'")
#                    params['tol_background_integration'] = 1.e-12
#                    params.pop('a_ini_over_a_today_default')
#                    params['background_evolver'] = 0
#                    params['background_integration_stepsize'] = 0.01
#                elif "error in perturbations_solve" in e.message and "condition (absh <= hmin) is true; Step size too small" in e.message and 'evolver' not in params:
#                    print("Changing 'evolver' to rk", end=' ')
##class_precision_parameter(perturbations_integration_stepsize,double,0.5)
##class_precision_parameter(perturbations_sampling_stepsize,double,0.1)
#                    params['evolver'] = 0
#                    params['perturbations_integration_stepsize'] = 0.1
#                    params['perturbations_sampling_stepsize'] = 0.01
                else:
                    raise classy.CosmoComputationError(e.message)
        print("succeeded =", succeeded, flush=True)

        # CMB power spectra
        if idx not in TT_array[:, 0] or idx not in TE_array[:, 0] or idx not in EE_array[:, 0] or idx not in PP_array[:, 0]:
            Cl = cosmo.lensed_cl(lmax)
        
        # TT
        if idx not in TT_array[:, 0]:
            assert np.all(np.diff(TT_array[:, 0]) > 0)
            arg = np.argwhere(idx < TT_array[:, 0]).ravel()[0]
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['tt'][2:]))
            TT_array = np.vstack((TT_array[:arg], [cosmo_array], TT_array[arg:]))

        # TE
        if idx not in TE_array[:, 0]:
            assert np.all(np.diff(TE_array[:, 0]) > 0)
            arg = np.argwhere(idx < TE_array[:, 0]).ravel()[0]
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['te'][2:]))
            TE_array = np.vstack((TE_array[:arg], [cosmo_array], TE_array[arg:]))

        # EE
        if idx not in EE_array[:, 0]:
            assert np.all(np.diff(EE_array[:, 0]) > 0)
            arg = np.argwhere(idx < EE_array[:, 0]).ravel()[0]
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['ee'][2:]))
            EE_array = np.vstack((EE_array[:arg], [cosmo_array], EE_array[arg:]))

        # PP
        if idx not in PP_array[:, 0]:
            assert np.all(np.diff(PP_array[:, 0]) > 0)
            arg = np.argwhere(idx < PP_array[:, 0]).ravel()[0]
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['pp'][2:]))
            PP_array = np.vstack((PP_array[:arg], [cosmo_array], PP_array[arg:]))
        
        # linear power spectrum
        if idx not in PKL_array[:, 0]:
            assert np.all(np.diff(PKL_array[:, 0]) > 0)
            arg = np.argwhere(idx < PKL_array[:, 0]).ravel()[0]
            P_lin = np.array([cosmo.pk_lin(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_lin))
            PKL_array = np.vstack((PKL_array[:arg], [cosmo_array], PKL_array[arg:]))

        # non linear power spectrum
        if idx not in PKNL_array[:, 0]:
            assert np.all(np.diff(PKNL_array[:, 0]) > 0)
            arg = np.argwhere(idx < PKNL_array[:, 0]).ravel()[0]
            P_nonlin = np.array([cosmo.pk(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_nonlin))
            PKNL_array = np.vstack((PKNL_array[:arg], [cosmo_array], PKNL_array[arg:]))

        # angular diameter distance
        if idx not in DAZ_array[:, 0]:
            assert np.all(np.diff(DAZ_array[:, 0]) > 0)
            arg = np.argwhere(idx < DAZ_array[:, 0]).ravel()[0]
            assert z_modes[0] == 0
            d_ang = np.array([cosmo.angular_distance(z) for z in z_modes[1:]])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], d_ang))
            DAZ_array = np.vstack((DAZ_array[:arg], [cosmo_array], DAZ_array[arg:]))

        # Hubble parameter
        if idx not in HZ_array[:, 0]:
            assert np.all(np.diff(HZ_array[:, 0]) > 0)
            arg = np.argwhere(idx < HZ_array[:, 0]).ravel()[0]
            Hubble = np.array([cosmo.Hubble(z) * c / 1e3 for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Hubble))
            HZ_array = np.vstack((HZ_array[:arg], [cosmo_array], HZ_array[arg:]))

        # sigma8 parameter
        if idx not in S8Z_array[:, 0]:
            assert np.all(np.diff(S8Z_array[:, 0]) > 0)
            arg = np.argwhere(idx < S8Z_array[:, 0]).ravel()[0]
            sigma8 = np.array([cosmo.sigma(R=8, z=z, h_units=True) for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], sigma8))
            S8Z_array = np.vstack((S8Z_array[:arg], [cosmo_array], S8Z_array[arg:]))

        # derived parameters
        if idx not in DER_array[:, 0]:
            assert np.all(np.diff(DER_array[:, 0]) > 0)
            arg = np.argwhere(idx < DER_array[:, 0]).ravel()[0]
            derived_dict = cosmo.get_current_derived_parameters(params_derived)
            derived_keys = list(derived_dict.keys())
            derived_values = np.array(list(derived_dict.values()))
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], derived_values))
            DER_array = np.vstack((DER_array[:arg], [cosmo_array], DER_array[arg:]))

    # parameter set rejected by Class
    except classy.CosmoComputationError as failure_message:
        #TT_array =   np.vstack((TT_array[:i],   [np.full_like(TT_array[0, :],   np.nan)], TT_array[i:]))
        #TE_array =   np.vstack((TE_array[:i],   [np.full_like(TE_array[0, :],   np.nan)], TE_array[i:]))
        #EE_array =   np.vstack((EE_array[:i],   [np.full_like(EE_array[0, :],   np.nan)], EE_array[i:]))
        #PP_array =   np.vstack((PP_array[:i],   [np.full_like(PP_array[0, :],   np.nan)], PP_array[i:]))
        #PKL_array =  np.vstack((PKL_array[:i],  [np.full_like(PKL_array[0, :],  np.nan)], PKL_array[i:]))
        #PKNL_array = np.vstack((PKNL_array[:i], [np.full_like(PKNL_array[0, :], np.nan)], PKNL_array[i:]))
        #DAZ_array =  np.vstack((DAZ_array[:i],  [np.full_like(DAZ_array[0, :],  np.nan)], DAZ_array[i:]))
        #HZ_array =   np.vstack((HZ_array[:i],   [np.full_like(HZ_array[0, :],   np.nan)], HZ_array[i:]))
        #S8Z_array =  np.vstack((S8Z_array[:i],  [np.full_like(S8Z_array[0, :],  np.nan)], S8Z_array[i:]))
        #DER_array =  np.vstack((DER_array[:i],  [np.full_like(DER_array[0, :],  np.nan)], DER_array[i:]))
        print()
        print(failure_message.message)
        print(f"params = {params}")
        print(cosmo.pars)
        print()
        cosmo.struct_cleanup()
        cosmo.empty()
        break

    # something wrong in Class init
    except classy.CosmoSevereError as critical_message:
        print("Something went wrong when calling CLASS" + str(critical_message))
        print(f"params = {params}")
        print(cosmo.pars)
        cosmo.struct_cleanup()
        cosmo.empty()
        break
    
    cosmo.struct_cleanup()
    cosmo.empty()
    if input_i >= 0:
        break

print()

np.savetxt(f"{path}/{cube}__{model}__TT_batch{batch:03d}.dat",   TT_array,   header=header['TT'])
np.savetxt(f"{path}/{cube}__{model}__TE_batch{batch:03d}.dat",   TE_array,   header=header['TE'])
np.savetxt(f"{path}/{cube}__{model}__EE_batch{batch:03d}.dat",   EE_array,   header=header['EE'])
np.savetxt(f"{path}/{cube}__{model}__PP_batch{batch:03d}.dat",   PP_array,   header=header['PP'])
np.savetxt(f"{path}/{cube}__{model}__PKL_batch{batch:03d}.dat",  PKL_array,  header=header['PKL'])
np.savetxt(f"{path}/{cube}__{model}__PKNL_batch{batch:03d}.dat", PKNL_array, header=header['PKNL'])
np.savetxt(f"{path}/{cube}__{model}__DAZ_batch{batch:03d}.dat",  DAZ_array,  header=header['DAZ'])
np.savetxt(f"{path}/{cube}__{model}__HZ_batch{batch:03d}.dat",   HZ_array,   header=header['HZ'])
np.savetxt(f"{path}/{cube}__{model}__S8Z_batch{batch:03d}.dat",  S8Z_array,  header=header['S8Z'])
np.savetxt(f"{path}/{cube}__{model}__DER_batch{batch:03d}.dat",  DER_array,  header=header['DER'])

os.remove(f"{path}/{cube}__{model}__TT_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__TE_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__EE_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__PP_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__PKL_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__PKNL_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__DAZ_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__HZ_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__S8Z_batch{batch:03d}_.npy")
os.remove(f"{path}/{cube}__{model}__DER_batch{batch:03d}_.npy")
