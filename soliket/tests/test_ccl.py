"""
Check that CCL works correctly.
"""

import copy
import importlib

import numpy as np
import pytest
from cobaya.likelihood import Likelihood
from cobaya.model import get_model


class CheckLike(Likelihood):
    """
    This is a mock likelihood that simply forces soliket.CCL to calculate
    a CCL object.
    """

    def logp(self, **params_values):
        _ = self.provider.get_CCL()
        return -1.0

    def get_requirements(self):
        return {"CCL": None}


ccl_like_and_theory = {
    "likelihood": {"checkLike": {"external": CheckLike}},
    "theory": {"camb": {}, "soliket.CCL": {"kmax": 10.0, "nonlinear": True}},
}


def test_ccl_import(check_skip_pyccl):
    """
    Test whether we can import pyCCL.
    """
    _ = importlib.import_module("pyccl")


def test_wrong_types(check_skip_pyccl):
    from soliket.ccl import CCL

    base_case = {"kmax": 0.1, "nonlinear": True, "z": 0.5, "extra_args": {}}
    wrong_type_cases = {
        "kmax": "not_a_float",
        "nonlinear": "not_a_bool",
        "z": "not_a_float_or_list",
        "extra_args": "not_a_dict",
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = CCL(**case)


def test_ccl_cobaya(check_skip_pyccl, evaluate_one_info, test_cosmology_params):
    """
    Test whether we can call CCL from cobaya.
    """
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(ccl_like_and_theory)

    model = get_model(evaluate_one_info)
    model.loglikes()


def test_ccl_distances(check_skip_pyccl, evaluate_one_info, test_cosmology_params):
    """
    Test whether the calculated angular diameter distance & luminosity distances
    in CCL have the correct relation.
    """
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(ccl_like_and_theory)

    model = get_model(evaluate_one_info)
    model.loglikes({})
    cosmo = model.provider.get_CCL()["cosmo"]

    z = np.linspace(0.0, 10.0, 100)
    a = 1.0 / (z + 1.0)

    da = cosmo.angular_diameter_distance(a)
    dl = cosmo.luminosity_distance(a)

    assert np.allclose(da * (1.0 + z) ** 2.0, dl)


def test_ccl_pk(check_skip_pyccl, evaluate_one_info, test_cosmology_params):
    """
    Test whether non-linear Pk > linear Pk in expected regimes.
    """
    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info.update(ccl_like_and_theory)

    model = get_model(evaluate_one_info)
    model.loglikes({})
    cosmo = model.provider.get_CCL()["cosmo"]

    k = np.logspace(np.log10(3e-1), 1, 1000)
    pk_lin = cosmo.linear_matter_power(k, a=0.5)
    pk_nonlin = cosmo.nonlin_matter_power(k, a=0.5)

    assert np.all(pk_nonlin > pk_lin)
