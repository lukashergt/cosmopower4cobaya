"""
:Synopsis: Managing CosmoPower emulators as Boltzmann code
:Author: Lukas Hergt based on the version by Matthieu Tristram
         (import cosmopower from A. Spurio Mancini and from B. Bolliet)

.. note::

   If you use this cosmological code, please cite it as:
   A. Spurio Mancini, D. Piras, J. Alsing, B. Joachimi, and M. P. Hobson,
   MNRAS, 511, 1771 (2022). doi:10.1093/mnras/stac064
   *CosmoPower: emulating cosmological power spectra for accelerated Bayesian
   inference from next-generation surveys*
   (`arXiv:2106.03846 <https://arxiv.org/abs/2106.03846>`_)

"""
import os
from os.path import join
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from cosmopower import cosmopower_NN as cp_NN, cosmopower_PCAplusNN as cp_PCA_NN

from cobaya.log import LoggedError
from cobaya.conventions import Const
from cobaya.theories.cosmo import BoltzmannBase

from __init__ import __version__


H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}
T_cmb = 2.7255  # in Kelvin (as fixed in CLASS)


class CosmoPower(BoltzmannBase):
    """CosmoPower cosmological Boltzmann emulator \\cite{SpurioMancini2022}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosmo_model = self.extra_args["cosmo_model"]
        self.TT_emu_name = self.extra_args.get("TT_emu_name", "TTTEEE/TT_v0")
        self.TE_emu_name = self.extra_args.get("TE_emu_name", "TTTEEE/TE_v0")
        self.EE_emu_name = self.extra_args.get("EE_emu_name", "TTTEEE/EE_v0")
        self.PP_emu_name = self.extra_args.get("PP_emu_name", "PP/PP_v0")
        self.DAZ_emu_name = self.extra_args.get("DAZ_emu_name", "growth-and-distances/DAZ_v0")
        self.DER_emu_name = self.extra_args.get("DER_emu_name", "derived-parameters/DER_v0")
        if not hasattr(self, 'python_path'):
            raise LoggedError(self.log, "If you want to use `cosmopower4cobaya` you need to "
                                        "specify the `python_path` in the input yaml file.")
        emu_path = os.path.normpath(join(self.python_path, "../cosmo_models", self.cosmo_model))
        self.log.info(f"Running with the emulator located at: {emu_path}")

        # CMB spectra
        self.TT_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.TT_emu_name))
        self.EE_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.EE_emu_name))
        self.TE_emu = cp_PCA_NN(restore=True, restore_filename=join(emu_path, self.TE_emu_name))
        self.PP_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.PP_emu_name))

        # matter power spectra
        # self.PKL_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.PKL_emu_name))
        # self.PKNL_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.PKNL_emu_name))

        # redshift dependent quantities
        self.DAZ_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.DAZ_emu_name))
        # self.HZ_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.HZ_emu_name))
        # self.S8Z_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.S8Z_emu_name))

        # derived parameters
        self.DER_emu = cp_NN(restore=True, restore_filename=join(emu_path, self.DER_emu_name))

        self.outputs = []

    def initialize(self):
        super().initialize()
        self.log.info(f"`cosmopower4cobaya` version {__version__} initialized.")

    def get_version(self):
        return __version__

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood
        super().must_provide(**requirements)
        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl":
                self.set_cl_reqs(v)
            elif k == "angular_diameter_distance":
                if 'DAZ' not in self.outputs:
                    self.outputs.append('DAZ')
            elif v is None:
                self._must_provide[k] = None
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Cleanup of products string
        self.check_no_repeated_input_extra()

    def set_cl_reqs(self, reqs):
        """Sets some common settings for lensed Cl's."""
        for XX in ["tt", "te", "ee", "pp"]:
            if XX not in self.outputs and any(XX in cl.lower() for cl in reqs):
                self.outputs.append(XX)

    def get_can_provide_params(self):
        """Get a list of derived parameters that this component can calculate.

        Returns
        -------
        names : list[str]
            List of derived parameters supported by the CosmoPower emulators.
        """
        names = [
            'Omega_m', 'sigma8',
            'z_eq', 'tau_eq', 'k_eq',
            'z_rec', 'rs_rec', 'ra_rec', 'tau_rec',
            'theta_s_100',
            'theta_star_100', 'z_star', 'rs_star', 'ra_star', 'tau_star',
            'z_d', 'rs_d',
            'z_reio',
            'YHe',
            'age', 'conformal_age'
        ]
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names

    def get_can_support_params(self):
        # non-exhaustive list of supported input parameters that will be assigned to
        # cosmopower4cobaya if they are varied
        if self.cosmo_model == 'LCDM':
            return ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'logA', 'n_s']
        elif self.cosmo_model == 'LCDM_OmegaK':
            return ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'logA', 'n_s', 'Omega_K']
        elif self.cosmo_model == 'LCDM_Mnu-d3':
            return ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'logA', 'n_s', 'M_nu']
        elif self.cosmo_model == 'wCDM':
            return ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'logA', 'n_s', 'w']
        elif self.cosmo_model == 'w0waCDM':
            return ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'logA', 'n_s', 'w', 'wa']

    # here modify if you want to bypass stuff in the class computation
    def calculate(self, state, want_derived=True, **params_values_dict):

        # Set parameters
        params_dict: dict[str, NDArray[np.float64]] = {}
        for k, v in params_values_dict.items():
            params_dict[self.translate_param(k)] = np.array([v])

        if 'tt' in self.outputs:
            state['tt'] = self.TT_emu.ten_to_predictions_np(params_dict)[0]
        if 'te' in self.outputs:
            state['te'] = self.TE_emu.predictions_np(params_dict)[0]
        if 'ee' in self.outputs:
            state['ee'] = self.EE_emu.ten_to_predictions_np(params_dict)[0]
        if 'pp' in self.outputs:
            state['pp'] = self.PP_emu.ten_to_predictions_np(params_dict)[0]
        if 'DAZ' in self.outputs:
            state['DAZ'] = self.DAZ_emu.ten_to_predictions_np(params_dict)[0]

        # compute derived parameters
        if want_derived:
            derived_params = self.get_can_provide_params()
            for p in self.output_params:
                if self.translate_param(p) not in derived_params:
                    raise LoggedError( self.log, f"Derived param not available: {p}")

            d = self.DER_emu.ten_to_predictions_np(params_dict)[0]
            state["derived"] = {p: d[derived_params.index(self.translate_param(p))]
                                for p in self.output_params}

        # for product, collector in self.collectors.items():
        #     # if product == 'fsigma8':
        #     #     method = getattr(self, collector.method)
        #     #     arg_array = self.collectors[product].arg_array
        #     #     if isinstance(arg_array, int):
        #     #         arg_array = np.atleast_1d(arg_array)
        #     #     if arg_array is None:
        #     #         print('dealing with arg_array none case... calling method')
        #     #     elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
        #     #         arg_array = np.array(arg_array)
        #     #         if len(arg_array.shape) == 1:
        #     #             # print('dealing with arg_array shape 1 case... calling method')
        #     #             # if more than one vectorised arg, assume all vectorised in parallel
        #     #             n_values = len(self.collectors[product].args[arg_array[0]])
        #     #             state[product] = np.zeros(n_values)
        #     #             args = deepcopy(list(self.collectors[product].args))
        #     #             args.append(params_values_dict)
        #     #             for i in range(n_values):
        #     #                 for arg_arr_index in arg_array:
        #     #                     args[arg_arr_index] = \
        #     #                         self.collectors[product].args[arg_arr_index][i]
        #     #                 state[product][i] = method(*args)
        #     #     else:
        #     #         raise LoggedError(self.log, "Variable over which to do an array call "
        #     #                                     f"not known: arg_array={arg_array}")
        #     if ('Pk_grid' in product) or ('comoving_radial_distance') in product:
        #         method = getattr(self, collector.method)
        #         args = deepcopy(list(self.collectors[product].args))
        #         args.append(params_values_dict)
        #         state[product] = method(*args)
        #         if collector.post:
        #             state[product] = collector.post(*state[product])

    # get the required new observable
    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        try:
            current_state = deepcopy(self.current_state)
        except:
            raise LoggedError(self.log, "Note that the CosmoPower emulators only provide lensed "
                                        "Cl's.")

        lmax = 5000
        ell = np.arange(lmax+1)
        ells_factor = ell * (ell + 1) / (2 * np.pi) if ell_factor else 1
        cls = {XX: np.zeros(lmax+1) for XX in self.outputs if XX in ['tt', 'te', 'ee', 'pp']}
        units_factor = self._cmb_unit_factor(units, T_cmb)

        for XX in cls:
            cls[XX][2:] = current_state[XX]
            units_power = float(sum(XX.count(x) for x in ['t', 'e', 'b']))
            cls[XX] *= units_factor**units_power
            if ell_factor:
                if XX == 'pp':
                    cls[XX] *= ells_factor**2 * (2 * np.pi)
                else:
                    cls[XX] *= ells_factor

        return cls

    def get_angular_diameter_distance(self, z):
        DAZ = deepcopy(self.current_state['DAZ'])
        Z = np.linspace(0, 20, 4001)[1:]
        Z2DA = interp1d(Z, DAZ, bounds_error=False, fill_value=np.nan)
        return Z2DA(z)

    # def get_param(self, p):
    #     translated = self.translate_param(p)
    #     if translated == 'rs_drag':
    #         return self.rs_drag()
    #
    #     if p == 'omegam':
    #         return self.Omega_m()

    # def rs_drag(self):
    #     return state["derived"]['rs_drag']

    # def Omega_m(self):
    #     params_values = self.current_state['params']
    #     result = (params_values['omega_b']+params_values['omega_cdm'])*(100./params_values['H0'])**2.
    #     return result

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", cls.__name__))
