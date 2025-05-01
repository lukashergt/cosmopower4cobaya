import sys
import numpy as np
from scipy import stats
import pyDOE as pyDOE

filepath_params = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM/LHC2e5__LCDM__parameter_file.npz'

path, filename_params = filepath_params.rsplit('/', 1)
cube = filename_params.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_params.split('__')[1]
assert model == 'w0waCDM'
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
w_limit = 0
w0_fld    = np.linspace(-3.0,  1.0,   n_samples)
wa_fld    = np.linspace(-15.0, 3+w_limit, n_samples)
# constraint at a=0: w(a) = w + wa*(1-a) < w_limit

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
    w0_fld,
    wa_fld,
    z,
])
n_params = AllParams.shape[0]
lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
idx = (lhd * n_samples).astype(int)
AllCombinations = np.zeros((n_samples, n_params))
for i in range(n_params):
    AllCombinations[:, i] = AllParams[i][idx[:, i]]

# Enforce constraint: w(a=0) = w + wa < 0
# to avoid behavious like matter (w=0) or radiation (w=1/3) at early times.
# Remove violating samples from latin hypercube.
# Redraw the corresponding number of lost samples uniformly (not latin hypercube).
idx_w = 6
idx_wa = 7
mask = np.argwhere(AllCombinations[:, idx_w] + AllCombinations[:, idx_wa] >= w_limit).ravel()
n_redraw = mask.size
w_min = w0_fld.min()
w_max = w0_fld.max()
wa_min = wa_fld.min()
wa_max = wa_fld.max()
area_tri = (wa_max-(w_limit-w_max)) * (w_max-w_min) / 2
area_rec = ((w_limit-w_max)-wa_min) * (w_max-w_min)
area_sum = area_tri + area_rec
redraw_tri = stats.dirichlet.rvs(alpha=np.ones(3), size=int(np.ceil(n_redraw * area_tri/area_sum)))
redraw_tri[:, 0] = redraw_tri[:, 0] * (w_max-w_min) + w_min
redraw_tri[:, 1] = redraw_tri[:, 1] * (wa_max-(w_limit-w_max)) + (w_limit-w_max)
redraw_rec_w = np.random.uniform(w_min, w_max,       int(np.floor(n_redraw * area_rec/area_sum)))
redraw_rec_wa = np.random.uniform(wa_min, (w_limit-w_max), int(np.floor(n_redraw * area_rec/area_sum)))
redraw_all = np.vstack((redraw_tri[:, :2], np.vstack((redraw_rec_w, redraw_rec_wa)).T))
np.random.shuffle(redraw_all)
AllCombinations[mask, idx_w:idx_wa+1] = redraw_all

# saving
params = {
    'ID': np.arange(n_samples),
    'omega_b': AllCombinations[:, 0],
    'omega_cdm': AllCombinations[:, 1],
    'H0': AllCombinations[:, 2],
    'tau_reio': AllCombinations[:, 3],
    'logA': AllCombinations[:, 4],
    'n_s': AllCombinations[:, 5],
    'w0_fld': AllCombinations[:, 6],
    'wa_fld': AllCombinations[:, 7],
    'z': AllCombinations[:, 8],
}

np.savez(filepath_params, **params)

