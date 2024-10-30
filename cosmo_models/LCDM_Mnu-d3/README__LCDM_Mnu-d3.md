# LCDM_Mnu-d3 v0
Suite of emulators for LCDM_Mnu-d3 (LCDM with 3 degenerate, massive neutrinos)


cosmopower version:
0.1.3


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
    'deg_ncdm': 3,
    'm_ncdm': 0.06,
    'N_ur': 0.00641,
    'omega_b':     params_LHC['omega_b'][idx],
    'omega_cdm':   params_LHC['omega_cdm'][idx],
    'H0':          params_LHC['H0'][idx],
    'tau_reio':    params_LHC['tau_reio'][idx],
    'ln_A_s_1e10': params_LHC['logA'][idx],
    'n_s':         params_LHC['n_s'][idx],
    'm_ncdm':      params_LHC['m_ncdm'][idx],
}


Package                      Version          
---------------------------- -----------------
classy                       3.2.3
Cython                       3.0.11
getdist                      1.5.3
keras                        2.15.0
matplotlib                   3.9.2
numpy                        1.24.3
nvidia-cublas-cu11           11.11.3.6
nvidia-cublas-cu12           12.3.4.1
nvidia-cuda-cupti-cu11       11.8.87
nvidia-cuda-cupti-cu12       12.3.101
nvidia-cuda-nvcc-cu11        11.8.89
nvidia-cuda-nvcc-cu12        12.3.107
nvidia-cuda-nvrtc-cu11       11.8.89
nvidia-cuda-nvrtc-cu12       12.3.107
nvidia-cuda-runtime-cu11     11.8.89
nvidia-cuda-runtime-cu12     12.3.101
nvidia-cudnn-cu11            9.5.0.50
nvidia-cudnn-cu12            8.9.7.29
nvidia-cufft-cu11            10.9.0.58
nvidia-cufft-cu12            11.0.12.1
nvidia-curand-cu11           10.3.0.86
nvidia-curand-cu12           10.3.4.107
nvidia-cusolver-cu11         11.4.1.48
nvidia-cusolver-cu12         11.5.4.101
nvidia-cusparse-cu11         11.7.5.86
nvidia-cusparse-cu12         12.2.0.103
nvidia-nccl-cu11             2.21.5
nvidia-nccl-cu12             2.19.3
nvidia-nvjitlink-cu12        12.3.101
pyDOE                        0.3.8
scipy                        1.14.1
tensorboard                  2.15.2
tensorboard-data-server      0.7.2
tensorflow                   2.15.1
tensorflow-estimator         2.15.0
tensorflow-io-gcs-filesystem 0.37.1
tensorflow-probability       0.22.1
tensorrt                     10.5.0
tensorrt-cu12                10.5.0
tensorrt-cu12-bindings       10.5.0
tensorrt-cu12-libs           10.5.0
tf_keras                     2.17.0

