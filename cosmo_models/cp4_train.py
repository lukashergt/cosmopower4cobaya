import sys
import os
import tensorflow as tf
tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import numpy as np
from cosmopower import cosmopower_NN, cosmopower_PCA, cosmopower_PCAplusNN

filepath_feature = str(sys.argv[1])  # e.g. 'LHC2e5__LCDM_Mnu-d3/LHC2e5__LCDM_Mnu-d3__TT_logpower.npz'
n_pcas = 512

path, filename_feature = filepath_feature.rsplit('/', 1)
cube = filename_feature.split('__')[0]
assert cube[:3] == 'LHC'
model = filename_feature.split('__')[1]
assert model in ['LCDM', 'LCDM_Mnu-d3', 'LCDM_OmegaK', 'wCDM', 'w0waCDM']
obs = filename_feature.split('__')[2].split('_')[0]
assert obs in ['TT', 'TE', 'EE', 'PP', 'PKL', 'PKNL', 'DAZ', 'HZ', 'S8Z', 'DER']
assert path == f"{cube}__{model}"
assert filepath_feature == f"{path}/{cube}__{model}__{obs}_logpower.npz"

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print()
print(f"running {obs} for {model} on {device} device using {cube}")

# training parameters
training_parameters = np.load(f"{path}/{cube}__{model}__{obs}_params.npz")
print(f"training parameters: {training_parameters.files}")

# training features
if obs == 'TE':
    training_features = np.load(f"{path}/{cube}__{model}__{obs}_power.npz")
else:
    training_features = np.load(f"{path}/{cube}__{model}__{obs}_logpower.npz")
modes = training_features['modes']
features = training_features['features']
print(f"number samples, number modes = {features.shape}")

#with tf.device(device):
#with tf.device('cpu'):
if obs == 'TE':
    cp_pca = cosmopower_PCA(
        parameters=list(training_parameters.files),
        modes=modes,
        n_pcas=n_pcas,
        parameters_filenames=[f"{path}/{cube}__{model}__{obs}_params"],
        features_filenames=[f"{path}/{cube}__{model}__{obs}_power"],
        verbose=True, # useful to follow the various steps
    )
    cp_pca.transform_and_stack_training_data()
    cp_nn = cosmopower_PCAplusNN(
        cp_pca=cp_pca,
        n_hidden=[512, 512, 512, 512],  # 4 hidden layers, each with 512 nodes
        verbose=True, # useful to understand the different steps in initialisation and training
    )
    kwargs = {}
else:
    cp_nn = cosmopower_NN(
        parameters=list(training_parameters.files),
        modes=modes,
        n_hidden=[512, 512, 512, 512],
        verbose=True,
    )
    kwargs = dict(
        training_parameters=training_parameters,
        training_features=features,
    )

if obs in ['PKL', 'PKNL', 'DER']:
    kwargs['device'] = 'gpu'
elif obs == 'TE':
    pass
else:
    kwargs['device'] = 'cpu'

cp_nn.train(
    filename_saved_model=f"{path}/{cube}__{model}__{obs}_cp_NN",
    # cooling schedule
    validation_split=0.05,
    learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    batch_sizes=[1024] * 5,
    gradient_accumulation_steps=[1] * 5,
    # early stopping set up
    patience_values=[100] * 5,
    max_epochs=[1000] * 5,
    **kwargs
)
