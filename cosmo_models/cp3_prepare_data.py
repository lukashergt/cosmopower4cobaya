import sys
import numpy as np

filepath_params = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM/LHC2e5__LCDM__parameter_file.npz'
batch_size = 1000

path, filename_params = filepath_params.rsplit('/', 1)
cube = filename_params.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_params.split('__')[1]
assert model in ['LCDM', 'LCDM_Mnu-d3', 'LCDM_OmegaK', 'wCDM', 'w0waCDM']
n_samples = cube[3:]
assert n_samples in ['1e4', '2e5']
n_samples = int(float(n_samples))
assert n_samples == 10000 or n_samples == 200000
n_batches = n_samples // batch_size
assert n_batches == 10 or n_batches == 200
assert path == f"{cube}__{model}"
assert filepath_params == f"{path}/{cube}__{model}__parameter_file.npz"

params_LHC = dict(np.load(filepath_params))
params_LHC_keys = np.array(list(params_LHC.keys()))
n_params = params_LHC_keys.size
if model == 'LCDM':
    assert n_params == 8, params_LHC_keys
elif model in ['LCDM_Mnu-d3', 'LCDM_OmegaK', 'wCDM']:
    assert n_params == 9, params_LHC_keys
elif model == 'w0waCDM':
    assert n_params == 10, params_LHC_keys
assert params_LHC['ID'].size == n_samples
print(f"{filepath_params}, model={model}, n_samples={n_samples}", flush=True)

k_modes = np.loadtxt(f"{path}/{cube}__{model}__k_modes.txt")
z_modes = np.loadtxt(f"{path}/{cube}__{model}__z_modes.txt")
l_modes = np.loadtxt(f"{path}/{cube}__{model}__l_modes.txt").astype(int)
kmax = k_modes[-1]
zmax = z_modes[-1]
lmax = l_modes[-1]
params_derived = list(np.loadtxt(f"{path}/{cube}__{model}__derived_params.txt", dtype=str))

observables = ['DER', 'TT', 'TE', 'EE', 'PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z']
modes = {
    'TT': l_modes,
    'TE': l_modes,
    'EE': l_modes,
    'PP': l_modes,
    'PKL': k_modes,
    'PKNL': k_modes,
    'DAZ': z_modes[1:],
    'HZ': z_modes,
    'S8Z': z_modes,
    'DER': params_derived,
}

for obs in observables:
    print(obs, end=' ')
    # load predictions from Class: concatenations of parameters and spectra
    cosmo_array = np.loadtxt(f"{path}/{cube}__{model}__{obs}_batch000.dat")
    for batch in range(1, n_batches):
        print(batch, end=' ', flush=True)
        cosmo_array = np.vstack((cosmo_array, np.loadtxt(f"{path}/{cube}__{model}__{obs}_batch{batch:03d}.dat")))
    print()

    # assert there are no NaNs
    assert np.isfinite(cosmo_array).all(), np.argwhere(~np.isfinite(cosmo_array))

    params_LHC = dict(np.load(filepath_params))
    notmissing = np.searchsorted(params_LHC['ID'], cosmo_array[:, 0])
    mask = np.zeros(params_LHC['ID'].size, dtype=bool)
    mask[notmissing] = True
    print("delete missing:", params_LHC['ID'][~mask])
    for key in params_LHC.keys():
        params_LHC[key] = params_LHC[key][mask]
        #params_LHC[key] = np.delete(params_LHC[key], missing)

    # assert the parameter ordering and the parameter set ordering match
    print(params_LHC['ID'].size)
    print(cosmo_array[:, 0].size)
    assert params_LHC['ID'].size == cosmo_array[:, 0].size, print(params_LHC['ID'].size, cosmo_array[:, 0].size)
    #assert np.all(params_LHC['ID'] == np.arange(params_LHC['ID'].size)), print(params_LHC['ID'])
    #assert np.all(cosmo_array[:, 0] == np.arange(cosmo_array[:, 0].size)), print(np.argwhere(cosmo_array[:, 0] != np.arange(cosmo_array[:, 0].size)))
    assert np.all(cosmo_array[:, 0] == params_LHC['ID'])
    if obs == 'TE':
        cosmo_feature = cosmo_array[:, n_params:]
    else:
        assert np.all(cosmo_array[:, n_params:] > 0)
        cosmo_feature = np.log10(cosmo_array[:, n_params:])
    cosmo_feature_dict = {'modes': modes[obs], 'features': cosmo_feature}
    
    # save
    params_LHC.pop('ID')
    if obs not in ['PKL', 'PKNL']:
        params_LHC.pop('z')
    if obs in ['PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z']:
        params_LHC.pop('tau_reio')
    np.savez(f"{path}/{cube}__{model}__{obs}_params.npz", **params_LHC)
    if obs != 'TE':
        np.savez(f"{path}/{cube}__{model}__{obs}_logpower.npz", **cosmo_feature_dict)
    else:
        np.savez(f"{path}/{cube}__{model}__{obs}_power.npz", **cosmo_feature_dict)

