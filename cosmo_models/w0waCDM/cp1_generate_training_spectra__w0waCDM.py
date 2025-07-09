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

k_modes = np.logspace(-4, 1, 501)
z_modes = np.linspace(0, 20, 4001)
l_modes = np.arange(2, 5001, 1, dtype=int)
kmax = k_modes[-1]
zmax = z_modes[-1]
lmax = l_modes[-1]
params_derived = [
    'Omega_m',
    #'Omega_r',
    #'Omega_nu',
    #'Omega_Lambda',
    'Neff',
    'sigma8',
    #'S8',
    'YHe',
    'z_eq',
    'tau_eq',
    'k_eq',
    'z_rec',
    'tau_rec',
    'rs_rec',
    'ra_rec',
    'z_star',
    'tau_star',
    'rs_star',
    'ra_star',
    'theta_star_100',
    'theta_s_100',
    'z_d',
    'tau_d',
    'rs_d',
    'z_reio',
    'age',
    'conformal_age',
]
if batch == 0:
    np.savetxt(f"{path}/{cube}__{model}__k_modes.txt", k_modes)
    np.savetxt(f"{path}/{cube}__{model}__z_modes.txt", z_modes)
    np.savetxt(f"{path}/{cube}__{model}__l_modes.txt", l_modes, fmt='%d')
    np.savetxt(f"{path}/{cube}__{model}__derived_params.txt", params_derived, fmt='%s')

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
for obs in observables:
    if not os.path.exists(f'{path}/{cube}__{model}__{obs}_batch{batch:03d}.dat'):
        np.savetxt(f'{path}/{cube}__{model}__{obs}_batch{batch:03d}.dat', [], header=header[obs])
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

for i in range(batch_size):
    if input_i >= 0:
        i = input_i
    idx = batch*batch_size+i
    if (TT_array.size != 0 and TE_array.size != 0 and EE_array.size != 0 and PP_array.size != 0 and PKL_array.size != 0 and PKNL_array.size != 0 and DAZ_array.size != 0 and HZ_array.size != 0 and S8Z_array.size != 0 and DER_array.size != 0 and
        TT_array[-1, 0] >= idx and TE_array[-1, 0] >= idx and EE_array[-1, 0] >= idx and PP_array[-1, 0] >= idx and PKL_array[-1, 0] >= idx and PKNL_array[-1, 0] >= idx and DAZ_array[-1, 0] >= idx and HZ_array[-1, 0] >= idx and S8Z_array[-1, 0] >= idx and DER_array[-1, 0] >= idx):
        print(f"batch {batch}, param set {idx}, param ID {params_LHC['ID'][idx]}, already done", flush=True)
        if input_i >= 0:
            break
        else:
            continue
    elif TT_array.size != 0 and TE_array.size != 0 and EE_array.size != 0 and PP_array.size != 0 and PKL_array.size != 0 and PKNL_array.size != 0 and DAZ_array.size != 0 and HZ_array.size != 0 and S8Z_array.size != 0 and DER_array.size != 0:
        print(f"batch {batch}, param set {idx}, param ID {params_LHC['ID'][idx]},", end=" ", flush=True)
        print("highest feature ID", np.array([TT_array[-1, 0], TE_array[-1, 0], EE_array[-1, 0], PP_array[-1, 0], PKL_array[-1, 0], PKNL_array[-1, 0], DAZ_array[-1, 0], HZ_array[-1, 0], S8Z_array[-1, 0], DER_array[-1, 0]]).astype(int), end=" ", flush=True)
        print([round(params_LHC[key][idx], 5) for key in params_LHC_keys], flush=True)
    else:
        print(f"batch {batch}, param set {idx}, param ID {params_LHC['ID'][idx]}", flush=True)

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

    # Set the parameters to the cosmological code
    cosmo = classy.Class()
    cosmo.set(params)

    try:
        cosmo.compute()

        # CMB power spectra
        Cl = cosmo.lensed_cl(lmax)
        
        # TT
        if TT_array.size == 0 or TT_array[-1, 0] < idx:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['tt'][2:]))
            if TT_array.size == 0:
                TT_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__TT_batch{batch:03d}.dat', TT_array, header=f"{params_LHC_keys}, TT(ell)")
            else:
                TT_array = np.vstack((TT_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__TT_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # TE
        if TE_array.size == 0 or TE_array[-1, 0] < idx:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['te'][2:]))
            if TE_array.size == 0:
                TE_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__TE_batch{batch:03d}.dat', TE_array, header=f"{params_LHC_keys}, TE(ell)")
            else:
                TE_array = np.vstack((TE_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__TE_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # EE
        if EE_array.size == 0 or EE_array[-1, 0] < idx:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['ee'][2:]))
            if EE_array.size == 0:
                EE_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__EE_batch{batch:03d}.dat', EE_array, header=f"{params_LHC_keys}, EE(ell)")
            else:
                EE_array = np.vstack((EE_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__EE_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # PP
        if PP_array.size == 0 or PP_array[-1, 0] < idx:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['pp'][2:]))
            if PP_array.size == 0:
                PP_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__PP_batch{batch:03d}.dat', PP_array, header=f"{params_LHC_keys}, PP(ell)")
            else:
                PP_array = np.vstack((PP_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__PP_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # linear power spectrum
        if PKL_array.size == 0 or PKL_array[-1, 0] < idx:
            P_lin = np.array([cosmo.pk_lin(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_lin))
            if PKL_array.size == 0:
                PKL_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__PKL_batch{batch:03d}.dat', PKL_array, header=f"{params_LHC_keys}, P_lin(k)")
            else:
                PKL_array = np.vstack((PKL_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__PKL_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # non linear power spectrum
        if PKNL_array.size == 0 or PKNL_array[-1, 0] < idx:
            P_nonlin = np.array([cosmo.pk(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_nonlin))
            if PKNL_array.size == 0:
                PKNL_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__PKNL_batch{batch:03d}.dat', PKNL_array, header=f"{params_LHC_keys}, P_nonlin(k)")
            else:
                PKNL_array = np.vstack((PKNL_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__PKNL_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # angular diameter distance
        if DAZ_array.size == 0 or DAZ_array[-1, 0] < idx:
            assert z_modes[0] == 0
            d_ang = np.array([cosmo.angular_distance(z) for z in z_modes[1:]])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], d_ang))
            if DAZ_array.size == 0:
                DAZ_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__DAZ_batch{batch:03d}.dat', DAZ_array, header=f"{params_LHC_keys}, d_angular(z)")
            else:
                DAZ_array = np.vstack((DAZ_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__DAZ_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # Hubble parameter
        if HZ_array.size == 0 or HZ_array[-1, 0] < idx:
            Hubble = np.array([cosmo.Hubble(z) * c / 1e3 for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Hubble))
            if HZ_array.size == 0:
                HZ_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__HZ_batch{batch:03d}.dat', HZ_array, header=f"{params_LHC_keys}, H(z)")
            else:
                HZ_array = np.vstack((HZ_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__HZ_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # sigma8 parameter
        if S8Z_array.size == 0 or S8Z_array[-1, 0] < idx:
            sigma8 = np.array([cosmo.sigma(R=8, z=z, h_units=True) for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], sigma8))
            if S8Z_array.size == 0:
                S8Z_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__S8Z_batch{batch:03d}.dat', S8Z_array, header=f"{params_LHC_keys}, sigma8(z)")
            else:
                S8Z_array = np.vstack((S8Z_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__S8Z_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

        # derived parameters
        if DER_array.size == 0 or DER_array[-1, 0] < idx:
            derived_dict = cosmo.get_current_derived_parameters(params_derived)
            derived_keys = list(derived_dict.keys())
            derived_values = np.array(list(derived_dict.values()))
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], derived_values))
            if DER_array.size == 0:
                DER_array = np.array([cosmo_array])
                np.savetxt(f'{path}/{cube}__{model}__DER_batch{batch:03d}.dat', DER_array, header=f"{params_LHC_keys}, {derived_keys}")
            else:
                DER_array = np.vstack((DER_array, cosmo_array))
                with open(f"{path}/{cube}__{model}__DER_batch{batch:03d}.dat", 'a') as f:
                    np.savetxt(f, [cosmo_array])

    # parameter set rejected by Class
    except classy.CosmoComputationError as failure_message:
        print()
        print(failure_message.message)
        print(f"params = {params}")
        print(cosmo.pars)
        print()
        cosmo.struct_cleanup()
        cosmo.empty()

    # something wrong in Class init
    except classy.CosmoSevereError as critical_message:
        print("Something went wrong when calling CLASS" + str(critical_message))
        print(f"params = {params}")
        print(cosmo.pars)
        cosmo.struct_cleanup()
        cosmo.empty()
    
    cosmo.struct_cleanup()
    cosmo.empty()
    if input_i >= 0:
        break

