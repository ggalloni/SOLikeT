import numpy as np
from cobaya.theory import Provider

from soliket import utils
from soliket.gaussian import GaussianLikelihood


class PSLikelihood(GaussianLikelihood):
    name: str = "TT"
    kind: str = "tt"
    lmax: int = 6000
    provider: Provider

    def get_requirements(self) -> dict:
        return {"Cl": {self.kind: self.lmax}}

    def _get_Cl(self) -> dict[str, np.ndarray]:
        return self.provider.get_Cl(ell_factor=True)

    def _get_theory(self, **params_values) -> np.ndarray:
        cl_theory = self._get_Cl()
        return cl_theory[self.kind][: self.lmax]


class BinnedPSLikelihood(PSLikelihood):
    binning_matrix_path: str = ""

    def initialize(self):
        self.binning_matrix = self._get_binning_matrix()
        self.bin_centers = self.binning_matrix.dot(
            np.arange(self.binning_matrix.shape[1])
        )
        super().initialize()

    @classmethod
    def binner(
        cls, ell: np.ndarray, cl_values: np.ndarray, bin_edges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return utils.binner(ell, cl_values, bin_edges)

    def _get_binning_matrix(self) -> np.ndarray:
        return np.loadtxt(self.binning_matrix_path)

    def _get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bin_centers, np.loadtxt(self.datapath)

    def _get_theory(self, **params_values) -> np.ndarray:
        cl_theory: dict[str, np.ndarray] = self._get_Cl()
        return self.binning_matrix.dot(cl_theory[self.kind][: self.lmax])
