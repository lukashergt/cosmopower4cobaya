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
batch_size = 1000

path, filename_params = filepath_params.rsplit('/', 1)
cube = filename_params.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_params.split('__')[1]
assert model == 'LCDM'
n_samples = cube[3:]
assert n_samples in ['1e4', '2e5']
n_samples = int(float(n_samples))
assert n_samples == 10000 or n_samples == 200000
assert path == f"{cube}__{model}"
assert filepath_params == f"{path}/{cube}__{model}__parameter_file.npz"

params_LHC = np.load(filepath_params)
params_LHC_keys = np.array(list(params_LHC.keys()))
n_params = params_LHC_keys.size
assert params_LHC['ID'].size == n_samples
print(f"{filepath_params}, model={model}, n_samples={n_samples}, batch={batch}", flush=True)

k_modes = np.loadtxt(f"{path}/{cube}__{model}__k_modes.txt")
z_modes = np.loadtxt(f"{path}/{cube}__{model}__z_modes.txt")
l_modes = np.loadtxt(f"{path}/{cube}__{model}__l_modes.txt").astype(int)
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
assert TT_array[-1, 0]   == batch * batch_size + 999, print(TT_array[-1, 0])
assert TE_array[-1, 0]   == batch * batch_size + 999, print(TE_array[-1, 0])
assert EE_array[-1, 0]   == batch * batch_size + 999, print(EE_array[-1, 0])
assert PP_array[-1, 0]   == batch * batch_size + 999, print(PP_array[-1, 0])
assert PKL_array[-1, 0]  == batch * batch_size + 999, print(PKL_array[-1, 0])
assert PKNL_array[-1, 0] == batch * batch_size + 999, print(PKNL_array[-1, 0])
assert DAZ_array[-1, 0]  == batch * batch_size + 999, print(DAZ_array[-1, 0])
assert HZ_array[-1, 0]   == batch * batch_size + 999, print(HZ_array[-1, 0])
assert S8Z_array[-1, 0]  == batch * batch_size + 999, print(S8Z_array[-1, 0])
assert DER_array[-1, 0]  == batch * batch_size + 999, print(DER_array[-1, 0])
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
    idx = batch*batch_size+i
    if idx == TT_array[i, 0] == TE_array[i, 0] == EE_array[i, 0] == PP_array[i, 0] == PKL_array[i, 0] == PKNL_array[i, 0] == DAZ_array[i, 0] == HZ_array[i, 0] == S8Z_array[i, 0] == DER_array[i, 0]:
        print(f"{idx:06d}done", end=' ', flush=True)
        continue
    else:
        print(flush=True)
        print(batch, np.int64([idx, TT_array[i, 0], TE_array[i, 0], EE_array[i, 0], PP_array[i, 0], PKL_array[i, 0], PKNL_array[i, 0], DAZ_array[i, 0], HZ_array[i, 0], S8Z_array[i, 0], DER_array[i, 0]]), flush=True)
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
        'N_ncdm': 1,
        'N_ur': 2.0328,
        'm_ncdm': 0.06,
        'omega_b':     params_LHC['omega_b'][idx],
        'omega_cdm':   params_LHC['omega_cdm'][idx],
        'H0':          params_LHC['H0'][idx],
        'tau_reio':    params_LHC['tau_reio'][idx],
        'ln_A_s_1e10': params_LHC['logA'][idx],
        'n_s':         params_LHC['n_s'][idx],
    }
    z_idx = params_LHC['z'][idx]

    # Set the parameters to the cosmological code
    cosmo = classy.Class()
    cosmo.set(params)

    try:
        cosmo.compute()

        # CMB power spectra
        if TT_array[i, 0] != idx or TE_array[i, 0] != idx or EE_array[i, 0] != idx or PP_array[i, 0] != idx:
            Cl = cosmo.lensed_cl(lmax)
        
        # TT
        if TT_array[i, 0] == idx:
            pass
        else:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['tt'][2:]))
            TT_array = np.vstack((TT_array[:i], [cosmo_array], TT_array[i:]))

        # TE
        if TE_array[i, 0] == idx:
            pass
        else:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['te'][2:]))
            TE_array = np.vstack((TE_array[:i], [cosmo_array], TE_array[i:]))

        # EE
        if EE_array[i, 0] == idx:
            pass
        else:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['ee'][2:]))
            EE_array = np.vstack((EE_array[:i], [cosmo_array], EE_array[i:]))

        # PP
        if PP_array[i, 0] == idx:
            pass
        else:
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Cl['pp'][2:]))
            PP_array = np.vstack((PP_array[:i], [cosmo_array], PP_array[i:]))
        
        # linear power spectrum
        if PKL_array[i, 0] == idx:
            pass
        else:
            P_lin = np.array([cosmo.pk_lin(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_lin))
            PKL_array = np.vstack((PKL_array[:i], [cosmo_array], PKL_array[i:]))

        # non linear power spectrum
        if PKNL_array[i, 0] == idx:
            pass
        else:
            P_nonlin = np.array([cosmo.pk(k=ki, z=z_idx) for ki in k_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], P_nonlin))
            PKNL_array = np.vstack((PKNL_array[:i], [cosmo_array], PKNL_array[i:]))

        # angular diameter distance
        if DAZ_array[i, 0] == idx:
            pass
        else:
            assert z_modes[0] == 0
            d_ang = np.array([cosmo.angular_distance(z) for z in z_modes[1:]])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], d_ang))
            DAZ_array = np.vstack((DAZ_array[:i], [cosmo_array], DAZ_array[i:]))

        # Hubble parameter
        if HZ_array[i, 0] == idx:
            pass
        else:
            Hubble = np.array([cosmo.Hubble(z) * c / 1e3 for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], Hubble))
            HZ_array = np.vstack((HZ_array[:i], [cosmo_array], HZ_array[i:]))

        # sigma8 parameter
        if S8Z_array[i, 0] == idx:
            pass
        else:
            sigma8 = np.array([cosmo.sigma(R=8, z=z, h_units=True) for z in z_modes])
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], sigma8))
            S8Z_array = np.vstack((S8Z_array[:i], [cosmo_array], S8Z_array[i:]))

        # derived parameters
        if DER_array[i, 0] == idx:
            pass
        else:
            derived_dict = cosmo.get_current_derived_parameters(params_derived)
            derived_keys = list(derived_dict.keys())
            derived_values = np.array(list(derived_dict.values()))
            cosmo_array = np.hstack(([params_LHC[param][idx] for param in params_LHC], derived_values))
            DER_array = np.vstack((DER_array[:i], [cosmo_array], DER_array[i:]))

    # parameter set rejected by Class
    except classy.CosmoComputationError as failure_message:
        print(str(failure_message)+'\n')
        print(cosmo.pars)
        cosmo.struct_cleanup()
        cosmo.empty()
        break

    # something wrong in Class init
    except classy.CosmoSevereError as critical_message:
        print("Something went wrong when calling CLASS" + str(critical_message))
        print(cosmo.pars)
        cosmo.struct_cleanup()
        cosmo.empty()
        break
    
    cosmo.struct_cleanup()
    cosmo.empty()

print(batch)

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
