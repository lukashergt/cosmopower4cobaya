import sys
import numpy as np

filepath_params = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM/LHC2e5__LCDM__parameter_file.npz'
batch_size = 1000
model = 'LCDM'
cube = 'LHC2e5'
n_samples = 200000
n_batches = n_samples // batch_size

path, filename_params = filepath_params.rsplit('/', 1)
params_LHC = np.load(filepath_params)
params_LHC_keys = np.array(list(params_LHC.keys()))
n_params = params_LHC_keys.size
assert filename_params.split('__')[0] == cube
assert filename_params.split('__')[1] == model
assert params_LHC['ID'].size == n_samples

k_modes = np.loadtxt(f"{path}/{cube}__{model}__k_modes.txt")
z_modes = np.loadtxt(f"{path}/{cube}__{model}__z_modes.txt")
l_modes = np.loadtxt(f"{path}/{cube}__{model}__l_modes.txt").astype(int)
kmax = k_modes[-1]
zmax = z_modes[-1]
lmax = l_modes[-1]
params_derived = list(np.loadtxt(f"{path}/{cube}__{model}__derived_params.txt", dtype=str))

observables = ['TT', 'TE', 'EE', 'PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z', 'DER']
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
    assert np.isfinite(cosmo_array).all()

    # assert the parameter ordering and the parameter set ordering match
    params_LHC = dict(np.load(filepath_params))
    assert params_LHC['ID'].size == cosmo_array[:, 0].size, print(params_LHC['ID'].size, cosmo_array[:, 0].size)
    assert np.all(params_LHC['ID'] == np.arange(params_LHC['ID'].size)), print(params_LHC['ID'])
    assert np.all(cosmo_array[:, 0] == np.arange(cosmo_array[:, 0].size)), print(np.argwhere(cosmo_array[:, 0] != np.arange(cosmo_array[:, 0].size)))
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
        params_LHC.pop('z_pk')
    if obs in ['PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z']:
        params_LHC.pop('tau_reio')
    np.savez(f"{path}/{cube}__{model}__{obs}_params.npz", **params_LHC)
    if obs != 'TE':
        np.savez(f"{path}/{cube}__{model}__{obs}_logpower.npz", **cosmo_feature_dict)
    else:
        np.savez(f"{path}/{cube}__{model}__{obs}_power.npz", **cosmo_feature_dict)

