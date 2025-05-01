import sys
import numpy as np
import pyDOE as pyDOE

filepath_params = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM/LHC2e5__LCDM__parameter_file.npz'

path, filename_params = filepath_params.rsplit('/', 1)
cube = filename_params.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_params.split('__')[1]
assert model == 'LCDM_OmegaK'
n_samples = cube[3:]
assert n_samples in ['1e4', '2e5']
n_samples = int(float(n_samples))
assert n_samples == 10000 or n_samples == 200000
assert path == f"{cube}__{model}"
assert filepath_params == f"{path}/{cube}__{model}__parameter_file.npz"

# parameter ranges
omega_b   = np.linspace(0.01,  0.026, n_samples)
omega_cdm = np.linspace(0.05,  0.30,  n_samples)
H0        = np.linspace(30.0,  100.0, n_samples)
tau_reio  = np.linspace(0.01,  0.20,  n_samples)
logA      = np.linspace(2.5,   3.5,   n_samples)
n_s       = np.linspace(0.8,   1.2,   n_samples)
Omega_k   = np.linspace(-0.15, 0.15,  n_samples)

z  = np.linspace(0.0, 5.0, n_samples)
c_min = 3.13   # np.linspace(2.0, 4.0, n_samples)
eta_0 = 0.603  # np.linspace(0.5, 1.0, n_samples)

# LHS grid
AllParams = np.vstack([
    omega_b, 
    omega_cdm, 
    H0, 
    tau_reio, 
    logA, 
    n_s, 
    Omega_k, 
    z,
])
n_params = AllParams.shape[0]
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)
AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# saving
params = {
    'ID': np.arange(n_samples),
    'omega_b': AllCombinations[:, 0],
    'omega_cdm': AllCombinations[:, 1],
    'H0': AllCombinations[:, 2],
    'tau_reio': AllCombinations[:, 3],
    'logA': AllCombinations[:, 4],
    'n_s': AllCombinations[:, 5],
    'Omega_k': AllCombinations[:, 6],
    'z': AllCombinations[:, 7],
}

np.savez(filepath_params, **params)

